begin;
alter table public.analyses add column if not exists attempts integer not null default 0;
alter table public.analyses add column if not exists next_attempt_at timestamptz not null default now();
alter table public.analyses add column if not exists worker_id uuid;
alter table public.analyses add column if not exists output_path text;
alter table public.analyses add column if not exists stage_timings jsonb not null default '{}'::jsonb;
alter table public.analyses add column if not exists preprocessing_version text;
create table if not exists public.worker_health (
 worker_id uuid primary key, seen_at timestamptz not null default now(),
 model_ready boolean not null default false, model_version text,
 memory_bytes bigint not null default 0
);
alter table public.worker_health enable row level security;
revoke all on public.worker_health from public, anon, authenticated, service_role;
grant select, insert, update on public.worker_health to service_role;

create or replace function public.claim_analysis(worker_token uuid)
returns setof public.analyses language sql security definer set search_path=public as $$
 update public.analyses set status='processing', started_at=now(), worker_id=worker_token,
 attempts=attempts+1, failure_category=null, completed_at=null
 where id=(select id from public.analyses where status='queued' and next_attempt_at<=now()
 and attempts<3 order by created_at for update skip locked limit 1) returning *;
$$;
revoke all on function public.claim_analysis(uuid) from public, anon, authenticated;
grant execute on function public.claim_analysis(uuid) to service_role;
-- Old workers must be stopped before upgrading. Retain the signature for rollback only.
revoke execute on function public.claim_analysis() from service_role;

create or replace function public.recover_stale_analyses() returns void
language sql security definer set search_path=public as $$
 update public.analyses set
 status=case when attempts>=3 then 'failed' else 'queued' end,
 failure_category='worker_interrupted', worker_id=null,
 completed_at=case when attempts>=3 then now() else null end,
 next_attempt_at=now()+interval '30 seconds'
 where status='processing' and started_at<now()-interval '15 minutes';
$$;
revoke all on function public.recover_stale_analyses() from public, anon, authenticated;
grant execute on function public.recover_stale_analyses() to service_role;

create or replace function public.worker_is_ready() returns boolean
language sql security definer set search_path=public as $$
 select exists(select 1 from public.worker_health
 where model_ready and seen_at>now()-interval '75 seconds');
$$;
revoke all on function public.worker_is_ready() from public;
grant execute on function public.worker_is_ready() to anon, authenticated, service_role;

create or replace function public.analysis_metrics() returns jsonb
language plpgsql security definer set search_path=public as $$
begin
 if coalesce(auth.jwt()->'app_metadata'->>'role','')<>'qxai_admin' then
 raise sqlstate 'PT403' using message='Administrator required'; end if;
 return (select jsonb_build_object(
 'queued',count(*) filter(where status='queued'),
 'processing',count(*) filter(where status='processing'),
 'complete',count(*) filter(where status='complete'),
 'failed',count(*) filter(where status='failed'),
 'average_seconds',avg(duration_seconds) filter(where status='complete'),
 'p95_seconds',percentile_cont(0.95) within group(order by duration_seconds) filter(where status='complete'),
 'oldest_queue_seconds',coalesce(extract(epoch from now()-min(created_at) filter(where status='queued')),0),
 'failures_last_24h',count(*) filter(where status='failed' and created_at>now()-interval '24 hours'),
 'ready_workers',(select count(*) from public.worker_health where model_ready and seen_at>now()-interval '75 seconds'),
 'worker_memory_bytes',(select coalesce(sum(memory_bytes),0) from public.worker_health where seen_at>now()-interval '75 seconds')
 ) from public.analyses);
end $$;
revoke all on function public.analysis_metrics() from public, anon;
grant execute on function public.analysis_metrics() to authenticated;
create or replace function public.enqueue_analysis(new_id uuid) returns public.analyses language plpgsql security definer set search_path=public as $$
declare r public.analyses;
begin
 if auth.uid() is null then raise exception 'authentication required'; end if;
 perform pg_advisory_xact_lock(424201);
 if (select count(*) from public.analyses where status in ('queued','processing')) >= 100
 or (select count(*) from public.analyses where owner_id=auth.uid() and status in ('queued','processing')) >= 3 then
 raise sqlstate 'PT429' using message='Queue capacity reached'; end if;
 insert into public.analyses(id,owner_id) values(new_id,auth.uid()) returning * into r;
 return r;
end $$;
revoke all on function public.enqueue_analysis(uuid) from public, anon;
grant execute on function public.enqueue_analysis(uuid) to authenticated;

commit;

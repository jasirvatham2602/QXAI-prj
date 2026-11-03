create table public.analyses (
 id uuid primary key, owner_id uuid not null references auth.users(id) on delete cascade,
 created_at timestamptz not null default now(), started_at timestamptz, completed_at timestamptz,
 status text not null default 'queued' check (status in ('queued','processing','complete','failed')),
 result jsonb, duration_seconds double precision, model_version text, failure_category text
);
alter table public.analyses enable row level security;
-- Explicit grants support projects with automatic table exposure disabled.
grant usage on schema public to authenticated, service_role;
revoke all on table public.analyses from public, anon, authenticated, service_role;
grant select, delete on table public.analyses to authenticated;
grant select, insert, update, delete on table public.analyses to service_role;
create policy own_read on public.analyses for select to authenticated using (owner_id=auth.uid());
create policy own_delete on public.analyses for delete to authenticated using (owner_id=auth.uid() and status in ('complete','failed'));
create index analysis_jobs on public.analyses(status,created_at);
create function public.enqueue_analysis(new_id uuid) returns public.analyses language plpgsql security definer set search_path=public as $$
declare r public.analyses;
begin
 if auth.uid() is null then raise exception 'authentication required'; end if;
 perform pg_advisory_xact_lock(424201);
 if (select count(*) from public.analyses where status in ('queued','processing')) >= 100
 or (select count(*) from public.analyses where owner_id=auth.uid() and status in ('queued','processing')) >= 3 then
 raise exception 'queue capacity reached'; end if;
 insert into public.analyses(id,owner_id) values(new_id,auth.uid()) returning * into r;
 return r;
end $$;
revoke all on function public.enqueue_analysis(uuid) from public, anon;
grant execute on function public.enqueue_analysis(uuid) to authenticated;
create function public.claim_analysis() returns setof public.analyses language sql security definer set search_path=public as $$
 update public.analyses set status='processing', started_at=now()
 where id=(select id from public.analyses where status='queued' order by created_at for update skip locked limit 1)
 returning *;
$$;
revoke all on function public.claim_analysis() from public, anon, authenticated;
grant execute on function public.claim_analysis() to service_role;
insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types)
values ('analyses','analyses',false,8388608,array['image/png']);
create policy own_images_read on storage.objects for select to authenticated using(bucket_id='analyses' and (storage.foldername(name))[1]=auth.uid()::text);
create policy own_images_insert on storage.objects for insert to authenticated with check(bucket_id='analyses' and (storage.foldername(name))[1]=auth.uid()::text);
create policy own_images_delete on storage.objects for delete to authenticated using(bucket_id='analyses' and (storage.foldername(name))[1]=auth.uid()::text);
create function public.analysis_metrics() returns jsonb language plpgsql security definer set search_path=public as $$
begin
 if coalesce(auth.jwt()->'app_metadata'->>'role','') <> 'qxai_admin' then raise exception 'administrator required'; end if;
 return (select jsonb_build_object('queued',count(*) filter(where status='queued'),'processing',count(*) filter(where status='processing'),'complete',count(*) filter(where status='complete'),'failed',count(*) filter(where status='failed'),'average_seconds',avg(duration_seconds),'p95_seconds',percentile_cont(0.95) within group(order by duration_seconds)) from public.analyses);
end $$;
revoke all on function public.analysis_metrics() from public, anon;
grant execute on function public.analysis_metrics() to authenticated;

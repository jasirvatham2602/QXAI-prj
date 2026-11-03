begin;
alter table public.analyses add column if not exists original_filename text;
create or replace function public.enqueue_named_analysis(new_id uuid,upload_name text)
returns public.analyses language plpgsql security definer set search_path=public as $$
declare saved public.analyses;
begin
 if auth.uid() is null then raise sqlstate 'PT401' using message='Authentication required'; end if;
 if upload_name is not null and (length(upload_name)>255 or strpos(upload_name,'/')>0 or strpos(upload_name,chr(92))>0) then
 raise sqlstate 'PT422' using message='Invalid original filename'; end if;
 perform pg_advisory_xact_lock(424201);
 if (select count(*) from public.analyses where status in ('queued','processing'))>=100
 or (select count(*) from public.analyses where owner_id=auth.uid() and status in ('queued','processing'))>=3 then
 raise sqlstate 'PT429' using message='Queue capacity reached'; end if;
 insert into public.analyses(id,owner_id,original_filename) values(new_id,auth.uid(),nullif(upload_name,'')) returning * into saved;
 return saved;
end $$;
revoke all on function public.enqueue_named_analysis(uuid,text) from public,anon;
grant execute on function public.enqueue_named_analysis(uuid,text) to authenticated;
-- Keep old callers compatible without bypassing queue admission or ownership checks.
create or replace function public.enqueue_analysis(new_id uuid) returns public.analyses
language sql security invoker set search_path=public as $$
 select public.enqueue_named_analysis(new_id,null);
$$;
revoke all on function public.enqueue_analysis(uuid) from public,anon;
grant execute on function public.enqueue_analysis(uuid) to authenticated;
commit;

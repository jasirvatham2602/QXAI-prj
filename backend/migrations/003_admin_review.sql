begin;
create or replace function public.admin_analysis_history(page_limit integer default 50,page_offset integer default 0)
returns setof jsonb language plpgsql security definer set search_path=public as $$
begin
 if coalesce(auth.role(),'')<>'service_role' and coalesce(auth.jwt()->'app_metadata'->>'role','')<>'qxai_admin' then
 raise sqlstate 'PT403' using message='Administrator required'; end if;
 return query select to_jsonb(a)||jsonb_build_object('uploader',jsonb_build_object(
 'id',u.id,'email',u.email,'display_name',coalesce(u.raw_user_meta_data->>'full_name',u.raw_user_meta_data->>'name'),'joined_at',u.created_at))
 from public.analyses a left join auth.users u on u.id=a.owner_id
 order by a.created_at desc,a.id desc limit least(greatest(page_limit,1),100) offset greatest(page_offset,0);
end $$;
revoke all on function public.admin_analysis_history(integer,integer) from public,anon;
grant execute on function public.admin_analysis_history(integer,integer) to authenticated,service_role;

create or replace function public.admin_analysis_detail(analysis_id uuid)
returns jsonb language plpgsql security definer set search_path=public as $$
begin
 if coalesce(auth.role(),'')<>'service_role' and coalesce(auth.jwt()->'app_metadata'->>'role','')<>'qxai_admin' then
 raise sqlstate 'PT403' using message='Administrator required'; end if;
 return (select to_jsonb(a)||jsonb_build_object('uploader',jsonb_build_object(
 'id',u.id,'email',u.email,'display_name',coalesce(u.raw_user_meta_data->>'full_name',u.raw_user_meta_data->>'name'),'joined_at',u.created_at))
 from public.analyses a left join auth.users u on u.id=a.owner_id where a.id=analysis_id);
end $$;
revoke all on function public.admin_analysis_detail(uuid) from public,anon;
grant execute on function public.admin_analysis_detail(uuid) to authenticated,service_role;
create policy admin_images_read on storage.objects for select to authenticated
using(bucket_id='analyses' and coalesce(auth.jwt()->'app_metadata'->>'role','')='qxai_admin');
commit;

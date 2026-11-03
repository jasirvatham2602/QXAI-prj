-- Read-only test: a simulated ordinary JWT must not reach admin review functions.
begin transaction read only;
do $$
begin
 perform set_config('request.jwt.claims','{"role":"authenticated","sub":"ffffffff-ffff-4fff-8fff-ffffffffffff","app_metadata":{}}',true);
 perform set_config('request.jwt.claim.role','authenticated',true);
 perform set_config('request.jwt.claim.sub','ffffffff-ffff-4fff-8fff-ffffffffffff',true);
 execute 'set local role authenticated';
 begin
  perform public.admin_analysis_history(1,0);
  raise exception 'Ordinary user incorrectly accessed admin history';
 exception when sqlstate 'PT403' then null;
 end;
 begin
  perform public.admin_analysis_detail('00000000-0000-0000-0000-000000000001'::uuid);
  raise exception 'Ordinary user incorrectly accessed admin detail';
 exception when sqlstate 'PT403' then null;
 end;
 execute 'reset role';
end $$;
rollback;

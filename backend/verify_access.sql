-- Read-only RLS validation. No accounts or analysis records are created or deleted.
begin transaction read only;
do $$
declare owner uuid;
begin
 select owner_id into owner from public.analyses limit 1;
 if owner is null then raise exception 'An existing owner analysis is required for this check'; end if;
 if exists(select 1 from public.analyses where owner_id='ffffffff-ffff-4fff-8fff-ffffffffffff'::uuid) then
 raise exception 'Choose a different non-owner test identity'; end if;
 perform set_config('request.jwt.claim.sub',owner::text,true);
 execute 'set local role authenticated';
 if not exists(select 1 from public.analyses) then raise exception 'Owner cannot read own history'; end if;
 if exists(select 1 from public.analyses where owner_id<>auth.uid()) then raise exception 'Other histories are visible'; end if;
 if exists(select 1 from storage.objects where bucket_id='analyses' and (storage.foldername(name))[1]<>auth.uid()::text) then
 raise exception 'Other image metadata is visible'; end if;
 execute 'reset role';
 perform set_config('request.jwt.claim.sub','ffffffff-ffff-4fff-8fff-ffffffffffff',true);
 execute 'set local role authenticated';
 if exists(select 1 from public.analyses) then raise exception 'Non-owner can read history'; end if;
 if exists(select 1 from storage.objects where bucket_id='analyses') then raise exception 'Non-owner can read image metadata'; end if;
 execute 'reset role';
end $$;
rollback;

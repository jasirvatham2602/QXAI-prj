# QXAI release checks

## Application and models

- Run frontend tests, production build, relevant ESLint checks, and all backend tests.
- Verify actual sign-in/owner isolation, filename search and date/status filters, admin access denial for ordinary accounts, PDF/HTML/JSON exports, and private image delivery.
- Run `training/verify.py`; inspect the candidate manifest, frozen backbone buffers, class order, circuit, preprocessing, and held-out results.
- Keep both original weights. Select candidate weights and normalization as one atomic bundle; never mix model generations.
- Inspect a synthetic forward/saliency run before a worker reports ready.

## Existing free hosting

- Confirm the existing Hugging Face Docker Space is CPU Basic before applying deployment. Do not create or request paid hardware.
- Stage runtime files with `scripts/prepare_space.py`; no .env, training data, private scans, or backups may enter the staging directory.
- Store the worker secret only in the existing backend's server-side secret store. Netlify receives only the three public VITE settings.
- Confirm the host build succeeds and the remote worker passes self-test. Verify readiness after stopping the local worker so a local heartbeat cannot hide a remote failure.
- Publish the reviewed `dist/` to the existing Netlify site, or merge the reviewed source into its connected production branch after checks pass.
- Confirm `/admin` and `/about` routes load directly, the actual production API URL is used, and developer preview fixtures are disabled in production.
- Free hosting can sleep and cold-start. Do not claim an always-on SLA.

## Operations and privacy

- Confirm production disables local shared-password login and uses the verified administrator claim.
- Confirm expired admin access clears private rows/images in the interface.
- Keep records until users explicitly delete them. No scheduled cleanup is enabled.
- Verify an encrypted local application backup and offline recovery. Store its key separately and privately. A fresh-project Supabase disaster recovery is still a separate check.
- GitHub readiness monitoring uses the existing backend URL (optionally overridden by `QXAI_API_URL`) once the workflow is on the default branch. Normal failed-check notifications apply; no custom messages are sent.
- API liveness latency checks do not establish model throughput. No full inference-load or resilience benchmark has been performed.

## Rollback

Use previous Netlify/Hugging Face deployments; roll model weights and preprocessing back together. The additive database migrations preserve records. Do not drop tables/buckets or restore unauthenticated routes to roll back an application release.

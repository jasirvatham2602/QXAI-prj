# QXAI research workspace

QXAI combines a React dashboard, an authenticated FastAPI API, a background model worker, and Supabase authentication/database/private storage. It compares classical and simulated quantum model predictions, saves private histories, provides an administrator review panel, and exports PDF/HTML/JSON reports with the original scan filename.

## Hosting without paid services

The frontend uses the existing Netlify site, `https://qxai-fullstack-prj.netlify.app`. `netlify.toml` builds `dist/` and provides the SPA redirect and basic response headers. `public/_redirects` and `_headers` provide the same settings for manual deployments. Its three public configuration values are `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`, and `VITE_API_URL`. Never put a server secret in a VITE variable.

The existing Hugging Face Docker Space is `jasirvatham2602/qxai-backend-hugging-face`, running on CPU Basic. A single container supervises the API and one worker. No hardware upgrade, new compute Space, persistent disk purchase, or paid service is required by this configuration. The Space can sleep when inactive, so this is not an always-on availability guarantee. Opening the frontend contacts its readiness endpoint; a cold start may take time. Accepted jobs remain in Supabase and are recovered after a worker restart.

The runtime Dockerfile is `backend/Dockerfile.space`. Stage a reviewed bundle with:

```sh
python scripts/prepare_space.py /private/tmp/qxai-space-release --models models/candidates/cleaned-v3-seed42
python scripts/deploy_space.py /private/tmp/qxai-space-release
```

The second command only verifies the existing Space and bundle. `--apply` configures the existing backend's server-side secret/variables and uploads the runtime. It never requests paid hardware. Use an authenticated Hugging Face environment with `huggingface_hub` installed. Docker builds cannot be run locally on this machine because Docker is unavailable; the host build and readiness must be checked before considering a release deployed.

## Local development

- Frontend: `http://127.0.0.1:5174/`
- Admin review: `http://127.0.0.1:5174/admin`
- API: `http://127.0.0.1:7860`
- Design preview: `http://127.0.0.1:5174/?preview=1` (illustrative data; development only)

Install frontend dependencies with `npm ci`, then `npm run dev -- --host 127.0.0.1 --port 5174 --strictPort`. Install API dependencies from `backend/requirements-api.txt`; run `uvicorn main:app --host 127.0.0.1 --port 7860` from `backend/`. Install a compatible model environment separately and run `python worker.py` from `backend/`.

The ignored `.env.local` contains frontend public configuration. `backend/.env.local` contains API configuration. `backend/.env.worker.local` contains the server-only worker secret. `.env.production.local` contains public frontend production values only. Local files containing configuration are owner-readable only. See `.env.example` and `backend/.env.example` for names.

## Administrator access

The owner’s confirmed, verified account has trusted `app_metadata.role=qxai_admin` access. Sign out and back in to refresh its token. Administrators can review all submitted scans, predictions, timings, and uploader email/ID/optional name/account creation time. Review is read-only. Ordinary users have access only to their own histories and private images.

The requested `admin` / `password` credentials remain an explicitly enabled loopback-only development option. They use a salted hash, throttling, and a one-hour HttpOnly SameSite cookie. Production disables this shared-password option; it requires the verified administrator account. Expired/denied admin sessions clear private records from the interface.

## Database and access controls

All four migrations have already been applied to the configured Supabase project:

1. `20261103000100_initial.sql`
2. `20261103000200_reliability.sql`
3. `20261103000300_admin_review.sql`
4. `20261103000400_original_filename.sql`

Fresh projects may apply `supabase/migrations` in order. Do not reapply the initial schema to the existing project. Before enabling Supabase GitHub migration deployment, reconcile its migration history with the baseline already applied manually.

RLS protects histories and storage objects. The worker uses a server-only secret; production admin requests use verified user JWTs and trusted app metadata. Access checks in `backend/verify_access.sql` and `verify_admin_access.sql` use transaction-local claims and roll back without creating users or changing records. No external test accounts were created.

## Model training, selection, and provenance

The original checkpoints remain in `backend/` for rollback. New checkpoints, an exact split manifest, training histories, evaluation, and a model card are in `models/candidates/cleaned-v3-seed42/`. Follow `training/README.md` and `PROTOCOL_AUDIT.md` to reproduce actual local training from the requested cleaned Kaggle archive and the original project architecture.

The current cleaned-v3 dataset (official version 12) has no supplied test directory. Its 2,021 intended three-class images are split into 1,413 training, 201 validation, and 407 test images using the original 70/10/20 seed-42 protocol, with exact duplicate groups kept disjoint. The separate binary copies are excluded. Checkpoint selection uses validation macro F1; test metrics are reported afterward. The earlier full Kaggle dataset experiment preserves its supplied test split and remains recorded separately under `kaggle-v1-seed183`; its results are not a paired comparison with cleaned-v3. Subject identifiers are unavailable, so these are image-level research results and cannot establish patient-independent or clinical performance. Acquisition differences may confound PD classification. Historical training membership for the old checkpoints is unknown, and their source data overlaps some test images; their comparison is diagnostic only.

`QXAI_MODEL_DIR` explicitly chooses a candidate directory. `model_bundle.py` checks class order, image size, preprocessing, and both checkpoint hashes. Cleaned-v3 weights use RGB → bilinear 224×224 → ToTensor, without channel normalization, and preserve train-adapted BatchNorm buffers. The earlier v1 candidate requires ImageNet normalization. Legacy checkpoints also retain their original raw input. Both checkpoints and the preprocessing contract must change together. Saliency display reverses normalization for the background image.

`public/model-evaluation.json` contains aggregate metrics only. Training images, archive downloads, feature caches, private uploads, backups, and credentials are excluded from source/runtime release packaging. Dataset metadata does not establish unrestricted redistribution rights; see the model card for provenance and limitations.

## Reliability and monitoring

Workers run both models and saliency on a synthetic image before marking themselves ready. Heartbeats continue independently during inference. `/health/live` checks the API; `/health/ready` returns 503 if no recently checked worker is ready. API responses use `Cache-Control: no-store`.

The queue has atomic claims, per-user/global active-job limits, bounded transient retries, ten-minute inference deadlines, and abandoned-job recovery. Final writes are fenced by worker/attempt ownership. Each attempt has a unique output path. Explicit deletion removes the analysis's private objects and record. Failed startup self-tests prevent a worker from reporting ready.

Reports record model version, preprocessing, timing, filename, and images. Model scores are not calibrated clinical confidence. The synthetic test image checks processing only.

The GitHub checks workflow runs frontend tests/build/lint and backend tests. The operations workflow provides a daily readiness check against the existing backend URL (overridable with `QXAI_API_URL`), using normal GitHub failed-check notifications. It does not send custom messages or run cleanup. `scripts/load_check.py` performs a small read-only API responsiveness check; it is not an inference capacity benchmark.

## Retention and encrypted backup

**Keep scans and analysis history until users delete them.** No automatic deletion is configured. The optional legacy retention script is not scheduled and requires explicit `--apply` for any deletion.

Install `backend/requirements-ops.txt`. Create an encrypted backup on this computer:

```sh
python backend/backup.py create .qxai-private/backups/DATE.qxai.enc --key-file .qxai-private/backup.key
python backend/backup.py verify .qxai-private/backups/DATE.qxai.enc --key-file .qxai-private/backup.key
python backend/backup.py recover-local .qxai-private/backups/DATE.qxai.enc --key-file .qxai-private/backup.key --destination .qxai-private/recovery-DATE
```

A backup and local recovery were verified on November 3, 2026. Fernet authenticated encryption protects its records and referenced input/saliency files. The key and backup are owner-readable only and Git-ignored. Keep a separate secure copy of the key: losing it makes recovery impossible. No backup or key has been uploaded to another service.

This is an application-data backup of terminal records and referenced images, capped at 256 MB. It excludes auth accounts, credentials, active jobs, worker heartbeats, and obsolete attempt outputs. It is not a transactionally consistent database snapshot or an automated cloud backup. Recovery currently verifies and writes an offline private directory; a restore into Supabase must reconcile existing auth user IDs and apply schema/policies before importing records/objects. A fresh-project disaster recovery has not been exercised.

## Release checks and rollback

Run `npm test`, `npm run build`, the relevant ESLint checks, and `pytest backend/tests -q`. `training/verify.py` checks checkpoint/circuit/frozen-feature integrity. See `RELEASE_CHECKLIST.md`.

Keep previous Netlify and Hugging Face deployments. Roll back both model weights and preprocessing together. Do not drop tables or storage buckets for application rollback, and do not restore the original unauthenticated image-serving routes for private data.

Package reviewed sources and the frontend build with `python scripts/package_release.py /absolute/path/QXAI-release.zip`. Check that its contents exclude secrets, private backups, virtual environments, raw training images, and feature caches before sharing it.

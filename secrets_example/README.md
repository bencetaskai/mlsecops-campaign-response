# secrets_example

Placeholders only. These values are **not** used by anything.

Docker Compose reads the real values from `secrets/` (git-ignored), with the same file names:

```powershell
Copy-Item -Recurse secrets_example secrets   # then put your own values into secrets\*.txt
Remove-Item secrets\README.md
```

Each file holds exactly one value, without quotes. A trailing newline is fine.

`api_key.txt` is the key for `X-API-Key` on `/train` and `/predict`; the API refuses to start with fewer than 32
characters. Generate one with Python, which writes plain UTF-8 (PowerShell's `>` would write UTF-16):

```powershell
python -c "import pathlib, secrets; pathlib.Path('secrets/api_key.txt').write_text(secrets.token_urlsafe(32))"
```

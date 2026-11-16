# secrets_example

Placeholders only. These values are **not** used by anything.

Docker Compose reads the real values from `secrets/` (git-ignored), with the same file names:

```powershell
Copy-Item -Recurse secrets_example secrets   # then put your own values into secrets\*.txt
Remove-Item secrets\README.md
```

Each file holds exactly one value, without quotes. A trailing newline is fine.

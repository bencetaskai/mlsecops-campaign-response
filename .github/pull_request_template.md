Replace this paragraph with the merge commit body: what the branch delivers and why. PR title: "merge: <Summary>".
Do not use the merge button: it re-wraps the body at 72 characters and splits the Signed-off-by line. Write the full
message to a file, check it with tools/check_commit_msg.py, then merge with
gh pr merge <N> --merge --subject "merge: <Summary> (#<N>)" --body-file <body file>

References: MLS-

Signed-off-by: Bence Táskai <113670719+bencetaskai@users.noreply.github.com>

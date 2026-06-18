# c0 demo — why hybrid retrieval (RRF) beats keyword or vector alone
# Render:  vhs assets/demo-hybrid.tape

Output assets/demo-hybrid.gif

Set Shell "bash"
Set FontSize 15
Set Width 1180
Set Height 820
Set Padding 26
Set Theme "Catppuccin Mocha"
Set TypingSpeed 26ms

Hide
Type "export PATH=$HOME/.cargo/bin:$PATH"
Enter
Type "export PS1='\033[36m❯\033[0m '"
Enter
Type "clear"
Enter
Show

Sleep 700ms
Type "# One paraphrased question, three ways to retrieve. Zero keywords shared with 'Recharge':"
Sleep 700ms
Enter
Sleep 400ms

Type "# 1) keyword / BM25 — literal token match, misses the paraphrase:"
Sleep 400ms
Enter
Type 'c0 search "money charged every month" --keyword-only --limit 2'
Sleep 400ms
Enter
Sleep 2.5s

Type "# 2) vector — understands intent, finds Recharge with no shared words:"
Sleep 400ms
Enter
Type 'c0 search "money charged every month" --vector-only --limit 2'
Sleep 400ms
Enter
Sleep 2.5s

Type "# 3) hybrid / RRF — fuses both: exact-match precision + semantic recall:"
Sleep 400ms
Enter
Type 'c0 search "money charged every month" --limit 3'
Sleep 400ms
Enter
Sleep 5s

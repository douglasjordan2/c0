# c0 demo — the self-improving reflection loop: a dead end becomes a new concept
# Render:  vhs assets/demo-reflect.tape   (needs a working classifier; sources demo-reflect-setup.sh)

Output assets/demo-reflect.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 28
Set Theme "Catppuccin Mocha"
Set TypingSpeed 26ms

Hide
Type "export PATH=$HOME/.cargo/bin:$PATH"
Enter
Type "export PS1='\033[36m❯\033[0m '"
Enter
Type "bash assets/demo-reflect-setup.sh"
Enter
Sleep 1s
Type "clear"
Enter
Show

Sleep 700ms
Type "# c0 is asked about something it has no memory of:"
Sleep 600ms
Enter
Type 'c0 walk "biome javascript linter formatter"'
Sleep 500ms
Enter
Sleep 3s

Type "# The miss is queued. The reflection loop classifies it with an LLM:"
Sleep 600ms
Enter
Type "c0 reflector process"
Sleep 500ms
Enter
Sleep 13s

Type "# Verdict: a real, reusable concept. Commit it into the graph:"
Sleep 600ms
Enter
Type "c0 reflector apply"
Sleep 500ms
Enter
Sleep 3.5s

Type "# c0 just filled its own memory gap — no human in the loop."
Sleep 2.2s
Enter
Sleep 1.2s

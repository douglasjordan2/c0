# c0 demo — bi-temporal memory: same question, different point in time, different truth
# Render:  vhs assets/demo-temporal.tape   (sources assets/demo-temporal-setup.sh)

Output assets/demo-temporal.gif

Set Shell "bash"
Set FontSize 16
Set Width 1180
Set Height 720
Set Padding 28
Set Theme "Catppuccin Mocha"
Set TypingSpeed 26ms

Hide
Type "export PATH=$HOME/.cargo/bin:$PATH"
Enter
Type "export PS1='\033[36m❯\033[0m '"
Enter
Type "bash assets/demo-temporal-setup.sh"
Enter
Sleep 1s
Type "clear"
Enter
Show

Sleep 700ms
Type "# Knowledge has a timeline. In 2022, 'Next.js routing' meant the Pages Router."
Sleep 600ms
Enter
Type "# In 2023 the App Router replaced it — record the change WITH the date it happened:"
Sleep 700ms
Enter
Type 'c0 supersede "nextjs-routing" --with "nextjs-app-router" --as-of 2023-05-01'
Sleep 500ms
Enter
Sleep 3s

Type "# Ask the same question 'as of' 2022 — c0 returns the era-correct answer:"
Sleep 600ms
Enter
Type 'c0 walk "next.js routing" --as-of 2022-06-01'
Sleep 500ms
Enter
Sleep 3.5s

Type "# Ask today — c0 has moved on to the App Router (and still knows the lineage):"
Sleep 600ms
Enter
Type 'c0 walk "next.js routing"'
Sleep 500ms
Enter
Sleep 5s

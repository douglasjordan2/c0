# c0 demo — walk the graph, correct stale LLM knowledge
# Render:  vhs assets/demo.tape

Output assets/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1120
Set Height 660
Set Padding 28
Set Theme "Catppuccin Mocha"
Set TypingSpeed 28ms

# --- clean, minimal prompt (hidden setup) ---
Hide
Type "export PATH=$HOME/.cargo/bin:$PATH"
Enter
Type "export PS1='\033[36m❯\033[0m '"
Enter
Type "clear"
Enter
Show

Sleep 800ms
Type "# A pre-2026 LLM swears you create a Shopify app in Admin -> 'Develop apps'."
Sleep 900ms
Enter
Type "# That's wrong now. c0 holds the current truth — watch it walk the graph:"
Sleep 1s
Enter
Sleep 500ms

Type 'c0 walk "shopify app development" 2>/dev/null | head -26'
Sleep 600ms
Enter
Sleep 6s

Type "# hybrid retrieval -> graph traversal -> a patch that overrides stale training."
Sleep 2.2s
Enter
Sleep 1.2s

# Git Feature Branching (Pair Project)

## Basic workflow
```bash
git checkout -b feature/my-change
git add .
git commit -m "Implement change"
git push -u origin HEAD
```

## Open a pull request
- Use a descriptive title
- Summarize what changed and why
- Add a short test plan

## Merge strategy
- Prefer squash merge for clean history
- Delete the branch after merge

## Tips
- Keep branches small and focused
- Rebase main into your branch if it drifts

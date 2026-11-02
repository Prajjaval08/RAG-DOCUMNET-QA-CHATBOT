# Publish this project to GitHub

Repository target: `Prajjaval08/rag-document-qa-chatbot`.

## Option A — GitHub CLI (recommended)

1. Install Git and the GitHub CLI (`gh`).
2. Open a terminal in the project root.
3. Authenticate interactively:

   ```bash
   gh auth login
   gh auth status
   ```

4. Review `.gitignore` and confirm `.env` does not contain any API key that will be staged. Use `.env.example` for placeholders only.
5. Initialize the repo and publish it:

   ```bash
   git init
   git branch -M main
   git add .
   git status
   git commit -m "Initial commit: RAG document Q&A chatbot"
   gh repo create Prajjaval08/rag-document-qa-chatbot --public --source=. --remote=origin --push
   ```

If you already created the empty repository on GitHub, instead add its URL as a remote and push:

```bash
git remote add origin https://github.com/Prajjaval08/rag-document-qa-chatbot.git
git push -u origin main
```

If `origin` already exists, inspect it with `git remote -v` and update it only if it is incorrect. Do not overwrite a repository containing existing work.

## Option B — GitHub website

1. Create a new repository named `rag-document-qa-chatbot` under your account `Prajjaval08`.
2. Do not add a README, license, or `.gitignore` in the website wizard because this project already has them locally.
3. In the project root, run `git init`, `git add .`, `git commit -m "Initial commit: RAG document Q&A chatbot"`, `git branch -M main`, `git remote add origin https://github.com/Prajjaval08/rag-document-qa-chatbot.git`, and `git push -u origin main`.

## Before every push

- Run `git status` and inspect staged filenames.
- Ensure `.env`, API keys, real documents, `data/faiss_index/`, virtual environments, and model caches are not tracked.
- Never share a GitHub token or provider API key with an AI chat. Authenticate using the GitHub CLI or an approved GitHub connection.

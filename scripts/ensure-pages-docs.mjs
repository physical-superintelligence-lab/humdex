import { execSync } from 'node:child_process';

function run(command, options = {}) {
  return execSync(command, { stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8', ...options }).trim();
}

function parseRepoSlug(remoteUrl) {
  // Supports:
  // - git@github.com:owner/repo.git
  // - https://github.com/owner/repo.git
  const sshMatch = remoteUrl.match(/^git@github\.com:([^/]+\/[^/]+?)(?:\.git)?$/);
  if (sshMatch) return sshMatch[1];

  const httpsMatch = remoteUrl.match(/^https:\/\/github\.com\/([^/]+\/[^/]+?)(?:\.git)?$/);
  if (httpsMatch) return httpsMatch[1];

  return null;
}

try {
  const remoteUrl = run('git remote get-url origin');
  const repoSlug = parseRepoSlug(remoteUrl);

  if (!repoSlug) {
    console.log('[pages] Skip: origin is not a GitHub repository URL.');
    process.exit(0);
  }

  try {
    run('gh --version');
  } catch {
    console.log('[pages] Skip: GitHub CLI (gh) is not installed.');
    process.exit(0);
  }

  try {
    run('gh auth status');
  } catch {
    console.log('[pages] Skip: gh is not authenticated. Run: gh auth login');
    process.exit(0);
  }

  run(`gh api -X PUT repos/${repoSlug}/pages -f source[branch]=website -f source[path]=/docs`);
  console.log('[pages] Ensured GitHub Pages source is website/docs.');
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.log(`[pages] Skip: unable to update Pages source (${message}).`);
  process.exit(0);
}

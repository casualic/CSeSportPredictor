"""HLTV scraping functions.

These functions are designed to be called from Flask routes during
Claude Code sessions, with the actual scraping performed via MCP Playwright
tools. The scraped data is stored in the SQLite database.

For standalone use, each function returns data dicts that can be passed
to database.insert_prediction() or used to resolve predictions.
"""


def parse_odds(odds_str):
    """Parse odds string like '1.85' to float, return None if invalid."""
    if not odds_str:
        return None
    try:
        val = float(str(odds_str).strip())
        return val if val > 1.0 else None
    except (ValueError, TypeError):
        return None


def implied_probability(odds):
    """Convert decimal odds to implied probability."""
    if odds and odds > 1.0:
        return round(1.0 / odds, 4)
    return None


def compute_edge(model_prob, implied_prob):
    """Compute edge: model probability minus implied probability."""
    if model_prob is not None and implied_prob is not None:
        return round(model_prob - implied_prob, 4)
    return None


def parse_bo_format(text):
    """Extract BO format from text like 'Best of 3' or 'bo3'."""
    if not text:
        return "BO3"
    t = str(text).lower().strip()
    if "1" in t:
        return "BO1"
    if "5" in t:
        return "BO5"
    return "BO3"


# --------------------------------------------------------------------------
# JS snippets for MCP Playwright browser_evaluate
# --------------------------------------------------------------------------

EXTRACT_UPCOMING_JS = """
() => {
    const matches = [];
    document.querySelectorAll('.match-wrapper').forEach(el => {
        const teams = el.querySelectorAll('.match-teamname');
        if (teams.length < 2) return;
        const team1 = teams[0]?.textContent?.trim() || '';
        const team2 = teams[1]?.textContent?.trim() || '';
        if (!team1 || !team2) return;

        const link = el.closest('a') || el.querySelector('a') || (el.parentElement ? el.parentElement.closest('a') : null);
        const url = link ? link.href : '';
        const event = (el.querySelector('.match-event')?.textContent?.trim() || '').replace(/\\s+/g, ' ');
        const time = el.querySelector('.match-time')?.textContent?.trim() || '';
        const meta = el.querySelector('.match-meta')?.textContent?.trim() || '';

        // Extract date from parent match-day header
        let matchDate = '';
        const dayEl = el.closest('.match-day') || el.closest('[data-zonedgrouping-headline-classes]')?.parentElement;
        if (dayEl) {
            const headline = dayEl.querySelector('.match-day-headline, .standard-headline');
            if (headline) matchDate = headline.textContent?.trim() || '';
        }

        if (meta === 'Live' || !time) return;

        matches.push({team1, team2, url, event, time, meta, matchDate});
    });
    return JSON.stringify(matches);
}
"""

EXTRACT_MATCH_RESULT_JS = """
() => {
    const data = {players: [], maps: []};

    const teamNames = document.querySelectorAll('.teamName');
    data.team1 = teamNames[0]?.textContent?.trim() || '';
    data.team2 = teamNames[1]?.textContent?.trim() || '';

    data.event = document.querySelector('.event a')?.textContent?.trim() || '';
    data.date = document.querySelector('.timeAndEvent .date')?.textContent?.trim() || '';

    // Scores
    const teamScores = document.querySelectorAll('.team .won, .team .lost');
    if (teamScores.length >= 2) {
        data.score1 = parseInt(teamScores[0]?.textContent?.trim()) || 0;
        data.score2 = parseInt(teamScores[1]?.textContent?.trim()) || 0;
    }

    // Winner
    const wonEl = document.querySelector('.team .won');
    if (wonEl) {
        const teamEl = wonEl.closest('.team');
        data.winner = teamEl?.querySelector('.teamName')?.textContent?.trim() || '';
    }

    // Maps
    document.querySelectorAll('.mapholder').forEach(m => {
        const mapName = m.querySelector('.mapname')?.textContent?.trim();
        const scores = m.querySelectorAll('.results-team-score');
        if (mapName && scores.length >= 2) {
            const s1 = parseInt(scores[0]?.textContent?.trim()) || 0;
            const s2 = parseInt(scores[1]?.textContent?.trim()) || 0;
            data.maps.push({
                map_name: mapName,
                score1: s1, score2: s2,
                map_winner: s1 > s2 ? data.team1 : data.team2
            });
        }
    });

    // Player stats
    const statsContent = document.querySelector('.stats-content');
    if (statsContent) {
        const tables = statsContent.querySelectorAll('table.totalstats');
        tables.forEach(table => {
            const rows = table.querySelectorAll('tr');
            const teamName = rows[0]?.querySelector('.players')?.textContent?.trim() || '';
            for (let i = 1; i < rows.length; i++) {
                const row = rows[i];
                const playerCell = row.querySelector('.players');
                if (!playerCell) continue;
                const parts = playerCell.textContent.trim().split('\\n').map(s => s.trim()).filter(s => s);
                const alias = parts.length > 1 ? parts[parts.length - 1] : parts[0];
                const kdCell = row.querySelector('.kd.traditional-data');
                const kdText = kdCell?.textContent?.trim() || '0-0';
                const kdParts = kdText.split('-').map(s => parseInt(s.trim()));
                const adr = parseFloat(row.querySelector('.adr.traditional-data')?.textContent?.trim()) || 0;
                const kastText = row.querySelector('.kast.traditional-data')?.textContent?.trim() || '0%';
                const kast = parseFloat(kastText.replace('%', '')) || 0;
                const rating = parseFloat(row.querySelector('.rating')?.textContent?.trim()) || 0;
                data.players.push({
                    team: teamName, player: alias,
                    kills: kdParts[0] || 0, deaths: kdParts[1] || 0,
                    adr, kast, rating
                });
            }
        });
    }

    return JSON.stringify(data);
}
"""

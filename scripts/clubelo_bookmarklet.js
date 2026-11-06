// clubelo.com/ENG bookmarklet: copies 'date,name,elo' rows to the clipboard.
// Then on Nova (repo root):  venv/bin/python get_elo.py   and paste at the prompt
// (or save the rows to a file in the project and use  --file <path>).
//
// Build the bookmark URL (paste the output as a bookmark's URL):
//   grep -v '^//' scripts/clubelo_bookmarklet.js | node -e "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>console.log('javascript:'+encodeURIComponent(s.trim())))"
(() => {
  const looksLikeElo = text => {
    const value = parseFloat(text.replace(",", "."));
    return /^\d{3,4}([.,]\d+)?$/.test(text) && value >= 900 && value <= 2600;
  };
  let tableRows = [...document.querySelectorAll("table.sidebar-table tbody tr")];
  if (!tableRows.length) tableRows = [...document.querySelectorAll("tr")];

  const seenTeams = new Set();
  const teamElos = [];
  for (const row of tableRows) {
    const cellTexts = [...row.querySelectorAll("td")].map(cell => cell.innerText.trim());
    const eloIndex = cellTexts.findLastIndex(looksLikeElo);
    if (eloIndex < 1) continue;
    let teamName = cellTexts.slice(0, eloIndex).reverse().find(text => text && !/^\d+\.?$/.test(text));
    if (!teamName) continue;
    teamName = teamName.replace(/^\d+\.?\s+/, "");
    if (seenTeams.has(teamName)) continue;
    seenTeams.add(teamName);
    teamElos.push([teamName, cellTexts[eloIndex].replace(",", ".")]);
  }
  if (!teamElos.length) { alert("clubelo bookmarklet: no Elo table found. The page layout may have changed."); return; }

  const today = new Date().toLocaleDateString("en-CA");  // YYYY-MM-DD, local date
  const quoteCsv = text => '"' + text.replace(/"/g, '""') + '"';
  const csvText = "date,name,elo\n" + teamElos.map(([name, elo]) => `${today},${quoteCsv(name)},${elo}`).join("\n") + "\n";
  const summary = `${teamElos.length} teams` + (teamElos.length < 20 ? " (fewer than 20, check the table)" : "");

  const showForManualCopy = () => {
    const box = document.createElement("textarea");
    box.value = csvText;
    box.style.cssText = "position:fixed;top:10px;left:10px;width:420px;height:60vh;z-index:99999;font:12px monospace";
    box.ondblclick = () => box.remove();
    document.body.appendChild(box);
    box.select();
    const copied = document.execCommand && document.execCommand("copy");
    alert(copied ? `Copied ${summary}. (Double-click the box to close it.)`
                 : `Couldn't copy automatically: press Ctrl/Cmd+C now. ${summary}. (Double-click the box to close it.)`);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(csvText).then(() => alert(`Copied ${summary} to clipboard.`), showForManualCopy);
  } else {
    showForManualCopy();
  }
})();

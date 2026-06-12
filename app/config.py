from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
MATCH_RESULTS_FILE = DATA_DIR / "match_results.json"
UNAVAILABLE_FILE = DATA_DIR / "unavailable_players.json"

MATCHES_PER_PAGE = 12

COUNTRY_ISO = {
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr",
    "Czech Republic": "cz", "Canada": "ca", "Bosnia and Herzegovina": "ba",
    "United States": "us", "Paraguay": "py", "Qatar": "qa",
    "Switzerland": "ch", "Australia": "au", "Turkey": "tr",
    "Brazil": "br", "Morocco": "ma", "Haiti": "ht", "Scotland": "gb-sct",
    "Germany": "de", "Curacao": "cw", "Ivory Coast": "ci", "Ecuador": "ec",
    "Netherlands": "nl", "Japan": "jp", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Spain": "es", "Cape Verde": "cv", "Saudi Arabia": "sa", "Uruguay": "uy",
    "France": "fr", "Senegal": "sn", "Iraq": "iq", "Norway": "no",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "DR Congo": "cd", "Uzbekistan": "uz", "Colombia": "co",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}

GROUP_NAMES = [
    "Group A", "Group B", "Group C", "Group D", "Group E", "Group F",
    "Group G", "Group H", "Group I", "Group J", "Group K", "Group L",
]

TEAM_NAME_MAP = {"Curacao": "Curaçao"}


def normalize_team_name(team: str) -> str:
    return TEAM_NAME_MAP.get(team, team)


def flag_img(team: str, size: int = 20) -> str:
    iso = COUNTRY_ISO.get(team)
    if not iso:
        return ""
    return (
        f'<img src="https://flagcdn.com/w40/{iso}.png" '
        f'style="width:{size}px;height:auto;vertical-align:middle;border-radius:2px;" />'
    )

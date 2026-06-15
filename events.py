"""
events.py — Curated MH event calendars (PC/Steam dates).
Each event: (name, date, kind)  kind ∈ {dlc, title_update, collab, event}
Only Steam-relevant dates. Base-Rise Switch TUs (2.0/3.0, 2021) excluded —
Rise launched on Steam (Jan 2022) with all base content already included.
"""
EVENTS = {
    "World": [
        ("Witcher collab",   "2019-05-17", "collab"),        # Geralt/Leshen on PC
        ("Iceborne",         "2020-01-09", "dlc"),            # PC expansion
        ("TU1 Rajang",       "2020-02-06", "title_update"),
        ("TU2 Stygian/Safi", "2020-03-12", "title_update"),
        ("TU3 Raging/Furious","2020-04-09", "title_update"),
        ("TU4 Alatreon",     "2020-07-09", "title_update"),
        ("TU5 Fatalis",      "2020-10-01", "title_update"),   # final major update
    ],
    "Rise": [
        ("Sunbreak",         "2022-06-30", "dlc"),            # PC expansion
        ("SB TU1",           "2022-08-10", "title_update"),   # Seething Bazel, etc.
        ("SB TU2",           "2022-09-29", "title_update"),   # Flaming Espinas, etc.
        ("SB TU3 Chaotic Gore","2022-11-24","title_update"),
        ("SB TU4 Velkhana",  "2023-02-07", "title_update"),
        ("SB TU5 Amatsu",    "2023-04-20", "title_update"),
        ("SB Final Malzeno", "2023-06-08", "title_update"),   # Primordial Malzeno, finale
    ],
    "Wilds": [
        ("TU1 Mizutsune",    "2025-04-04", "title_update"),
        ("SF6 collab",       "2025-05-28", "collab"),         # Street Fighter 6 / Akuma
        ("TU2 Lagiacrus",    "2025-06-30", "title_update"),
        ("TU3 FFXIV/Omega",  "2025-09-29", "title_update"),   # Final Fantasy XIV collab
        ("TU4 Gogmazios",    "2025-12-16", "title_update"),
        ("AT Arkveld (Feb)", "2026-02-10", "title_update"),   # Feb 2026 update
        ("Expansion reveal", "2026-06-06", "event"),          # SGF reveal (approx)
    ],
}

# Support status per game:
#   "final_content" = date of last content drop (title update / DLC)
#   "status"        = "ended"  -> retired after final_content (exclude tail as drought)
#                     "active" -> still receiving / awaiting content (trailing dry = real)
SUPPORT = {
    "World": dict(final_content="2020-10-01", status="ended"),   # Fatalis TU5 (+AT Velkhana) = last
    "Rise":  dict(final_content="2023-06-08", status="ended"),   # Primordial Malzeno = finale
    "Wilds": dict(final_content="2026-02-10", status="active"),  # AT Arkveld last content; expansion announced, not out
}

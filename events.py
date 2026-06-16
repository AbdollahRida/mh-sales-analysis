"""
events.py — Curated MH event calendars (PC/Steam dates).
Each event: (name, date, kind)  kind ∈ {dlc, title_update, collab, event}
Only Steam-relevant dates. Base-Rise Switch TUs (2.0/3.0, 2021) excluded —
Rise launched on Steam (Jan 2022) with all base content already included.
"""
EVENTS = {
    "World": [
        # --- BASE MONSTER HUNTER: WORLD ERA ---
        ("Base Monster Hunter: World Console Launch (PS4 & Xbox One)", "2018-01-26", "release"),
        ("Title Update 1 (Ver. 2.00): Deviljho Added", "2018-03-22", "TU"),
        ("Ver. 3.00: Kulve Taroth Siege Quest & El Dorado Locale", "2018-04-19", "event"),
        ("Title Update 2 (Ver. 4.00): Lunastra Added", "2018-05-31", "TU"),
        ("Ver. 5.00: Final Fantasy XIV Collab (Behemoth Added)", "2018-08-02", "event"),
        ("Base Monster Hunter: World PC Launch (Steam)", "2018-08-09", "release"),
        ("Ver. 6.00: The Witcher 3: Wild Hunt Collab (Leshen Added)", "2019-02-08", "event"),
        ("Ver. 6.02: Arch-Tempered Nergigante (Final Base World Monster)", "2019-05-11", "patch"),

        # --- ICEBORNE EXPANSION ERA (Console Launch & Sync Cycle) ---
        ("Monster Hunter World: Iceborne Console Expansion Launch", "2019-09-06", "dlc"),
        ("Iceborne TU 1 (Ver. 11.00): Rajang & Volcanic Guiding Lands Zone", "2019-10-10", "TU"),
        ("Iceborne TU 2 (Ver. 12.00): Stygian Zinogre, Tundra Zone, & Safi'jiiva Recon", "2019-12-05", "TU"),
        ("Safi'jiiva Siege Quest Officially Opens", "2019-12-13", "event"),
        ("Monster Hunter World: Iceborne PC (Steam) Expansion Launch", "2020-01-09", "dlc"),
        ("Iceborne TU 3 (Ver. 13.00): Raging Brachydios & Furious Rajang", "2020-03-23", "TU"),
        ("Ver. 13.50: Master Rank Kulve Taroth & Namielle Arch-Tempered", "2020-04-23", "patch"),
        ("Iceborne TU 4 (Ver. 14.00): Alatreon Added (Delayed from May due to COVID-19)", "2020-07-09", "TU"),
        ("Iceborne TU 5 (Ver. 15.00): Fatalis, Arch-Tempered Velkhana (Final Content Patch)", "2020-10-01", "TU"),
        
        # --- POST-CONTENT MAINTENANCE & PERMANENCE ---
        ("Ver. 15.10: Artemis 'Monster Hunter Movie' Collab (Later Retired)", "2020-12-04", "event"),
        ("Ver. 15.11: Major Event Quest Cycle Automations (All festivals made permanent rotation)", "2021-01-01", "patch"),
        ("Steam Deck Compatibility / Modern OS Balance Adjustments", "2023-12-01", "patch")
    ],
    "Rise": [
        ("Base Monster Hunter Rise Nintendo Switch Launch", "2021-03-26", "release"),
        ("Update 2.0", "2021-04-28", "patch"),
        ("Update 3.0", "2021-05-27", "patch"),
        ("Update 3.1", "2021-06-24", "patch"),
        ("Update 3.2", "2021-07-29", "patch"),
        ("Update 3.3", "2021-08-26", "patch"),
        ("CAPCOM Collab 4: Megaman 11, Sunbreak Announced", "2021-09-24", "event"),
        ("Update 3.4", "2021-10-01", "patch"),
        ("Update 3.5", "2021-10-28", "patch"),
        ("CAPCOM Collab 5: Ghosts 'n Goblins Resurrection Collab Event", "2021-10-29", "event"),
        ("Update 3.6, Sega: Sonic 30th Anniversary Collab Event", "2021-11-25", "event"),
        ("Update 3.7", "2021-12-20", "patch"),
        ("PC Release", "2022-01-12", "release"),
        ("USJ Collab", "2022-01-21", "event"),
        ("Update 3.8 (Switch)", "2022-01-27", "patch"),
        ("Update 3.9 (Switch & PC)", "2022-02-24", "patch"),
        ("Update 3.10 (Switch)", "2022-03-31", "patch"),
        ("Update 10.0 (Sunbreak Expansion Launch)", "2022-06-30", "dlc"),
        ("Update 11.0 (Title Update 1)", "2022-08-10", "TU"),
        ("Update 12.0 (Title Update 2)", "2022-09-29", "TU"),
        ("Update 13.0 (Title Update 3)", "2022-11-24", "TU"),
        ("PlayStation & Xbox Base Game Launch", "2023-01-20", "release"),
        ("Update 14.0 (Title Update 4)", "2023-02-07", "TU"),
        ("Update 15.0 (Title Update 5)", "2023-04-20", "TU"),
        ("Update 16.0 / 16.0.2 (Bonus Update / Final Content Patch)", "2023-06-08", "TU")
    ],
    "Wilds": [
        ("Title Update 1 Ver 1.01", "2025-04-03", "TU"),
        ("Festival of Accord: Blossomdance", "2025-04-22", "event"),
        ("Arch-Tempered Rey Dau, Gamma Armor Sets", "2025-04-29", "event"),
        ("Free Challenge Quests", "2025-05-07", "event"),
        ("Version 1.011 Patch Notes", "2025-05-27", "patch"),
        ("Title Update 2 Ver 1.02", "2025-06-30", "TU"),
        ("Festival of Accord: Flamefete, Arch-tempered Uth Duna", "2025-07-22", "event"),
        ("Version 1.021/1.021.01/1.021.02", "2025-08-12", "patch"),
        ("Fender Special Collaboration", "2025-08-26", "event"),
        ("Title Update 3", "2025-09-29", "TU"),
        ("Title Update 4", "2025-12-15", "TU"),
        ("Version 1.041 Patch Notes", "2026-02-17", "patch")
    ]
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

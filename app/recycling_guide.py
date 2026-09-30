"""What to do with each predicted material.

Based on New Zealand's standard kerbside recycling rules (in force since
1 February 2024): glass bottles and jars; paper and cardboard; plastic bottles and
containers with recycling symbols 1, 2 and 5; aluminium and steel tins and cans.
Bin colours and collection details vary between councils, so the advice always
reminds the user to check their local council.
"""

GUIDE = {
    "cardboard": {
        "icon": "📦",
        "bin": "Recycling bin (paper & cardboard)",
        "recyclable": True,
        "tips": ["Flatten boxes and keep them dry.", "Remove plastic tape, film and polystyrene inserts.",
                 "Pizza boxes are accepted if food scraps are removed."],
    },
    "glass": {
        "icon": "🍾",
        "bin": "Glass crate / glass recycling",
        "recyclable": True,
        "tips": ["Only bottles and jars - no window glass, mirrors, drinking glasses or ceramics.",
                 "Rinse and remove lids (lids go in the rubbish)."],
    },
    "metal": {
        "icon": "🥫",
        "bin": "Recycling bin (aluminium & steel cans)",
        "recyclable": True,
        "tips": ["Rinse tins and cans.", "Aerosol cans, foil and foil trays are NOT accepted at kerbside.",
                 "Items smaller than 5 cm (e.g. bottle caps) go in the rubbish."],
    },
    "paper": {
        "icon": "📰",
        "bin": "Recycling bin (paper & cardboard)",
        "recyclable": True,
        "tips": ["Keep paper clean and dry.", "Shredded paper, tissues and paper towels are NOT accepted.",
                 "Liquid paperboard (juice/milk cartons) is NOT accepted at kerbside."],
    },
    "plastic": {
        "icon": "🧴",
        "bin": "Recycling bin (plastics 1, 2 and 5 only)",
        "recyclable": True,
        "tips": ["Check the number inside the recycling triangle: only 1, 2 and 5 are accepted.",
                 "Rinse containers and remove lids.", "Soft plastics (bags, wrap) are NOT accepted at kerbside."],
    },
    "trash": {
        "icon": "🗑️",
        "bin": "General rubbish (landfill)",
        "recyclable": False,
        "tips": ["Not recyclable at kerbside.", "Food scraps: use a food-scraps bin or compost where available.",
                 "Hazardous items (batteries, e-waste) need a special drop-off."],
    },
}

GENERAL_NOTE = "Bin colours and accepted items vary by council - check your local council's recycling guide."


def advice_for(label: str) -> dict:
    return GUIDE.get(label, {"icon": "❓", "bin": "Unknown", "recyclable": False, "tips": []})

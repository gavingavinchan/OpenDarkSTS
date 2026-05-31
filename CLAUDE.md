# Project Goal

Detect dark ship-to-ship (STS) transfers by first analyzing satellite imagery to identify likely STS events with a YOLO-based model, then cross-checking those detections against AIS records for matching time and location.

# Data Handling Rule

The folder `/home/gavin/Downloads/AIS data` is read-only. Agents must never modify it in any way, including writing files, changing files, moving files, renaming files, deleting files, or running code or commands that alter anything in that folder.

# Workspace Rule

Agents must do all working steps in their own workspace inside this repository and copy only the inputs they need from the read-only source when appropriate.

# Co-Authorship Rule

Never add Claude or any AI model (Claude Opus, Sonnet, GPT, etc.) as a co-author in git commits.

# Academic Writing Tone Rule

Unless explicitly told otherwise, treat all prose writing (the paper, draft sections, abstracts, captions, and similar) as text for the academic paper and write it in a formal, academic register. Use precise, impersonal phrasing; avoid contractions, colloquialisms, casual asides, marketing language, and analogies aimed at a general audience. This rule governs tone and wording only; it does not change the factual content the user asks for.

# Paper Audience Rule

Most readers of the paper want to learn how to use the tool; only a minority read it to understand how the tool is implemented. Write accordingly: lead with the user-facing overview (what the tool does and how it is operated), and place implementation detail after it or in clearly secondary positions. When a section opens, prefer an overview-first ordering, with whole-tool implementation notes following the user-perspective material.

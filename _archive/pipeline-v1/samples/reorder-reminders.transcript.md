# Normalized Transcript

- Source: `reorder-reminders.sample`
- Note: synthetic, information-dense sample for testing the requirement worksheet.

## Text

**Speaker A (PM):** Thanks everyone. The reason I pulled this group together is retention. We're good at getting someone to place a first order, but a huge chunk of customers buy once and we never see them again. Lifecycle pulled the numbers last month and it's not pretty — most first-time buyers never come back for a second order. I want us to leave today with a direction for a reorder nudge.

**Speaker B (Eng Lead):** When you say "never come back," do you mean across the whole catalog, or specifically the wines they bought before? Those are pretty different problems.

**Speaker A (PM):** Good question. I'm mostly thinking about the people who clearly loved something — bought a case, drank it — and just forgot to come back. So, favorites. Reorder the thing you already know you like.

**Speaker C (Data):** That framing matters for us. Wine isn't like razors or coffee where the reorder cadence is predictable. Someone might go through a case of an everyday red in three weeks, or sit on a nice Barolo for two years. If we guess the timing wrong we're going to annoy people.

**Speaker A (PM):** Right. My instinct is we send an email reminder thirty days after a purchase saying "running low? reorder your favorites." Simple.

**Speaker D (Lifecycle Marketing):** I'd push back gently on the thirty days. That number feels made up. We have purchase history — can't we actually look at how often a given customer reorders and personalize it?

**Speaker C (Data):** We could, eventually. We have order history in the orders service. But a per-customer cadence model is real work and I don't think we have it this quarter. We could start with a category-level heuristic — everyday wines get a shorter window, premium gets a longer one.

**Speaker B (Eng Lead):** Let me say the obvious constraint up front. Whatever we do, it has to go through Braze. We are not onboarding a new messaging vendor this quarter, that ship has sailed with procurement. So email and push are on the table because Braze does them. SMS is technically possible but we'd need legal to sign off on consent and I don't want to assume that.

**Speaker A (PM):** Okay, Braze it is for now. Honestly I was assuming email anyway.

**Speaker D (Lifecycle Marketing):** Email's fine but we have to be careful about frequency. These customers are already getting our weekly promo blasts and the new-vintage announcements. If reorder reminders stack on top of that with no cap, we'll spike unsubscribes, and deliverability takes months to recover once it dips. That's the thing that actually scares me here.

**Speaker B (Eng Lead):** Agreed, that's the real risk. A global frequency cap across all our Braze campaigns would solve it but we don't have one today. That might be a prerequisite, and it's not small.

**Speaker A (PM):** Noted. Let's flag that. What about the "subscribe" idea — should we just let people set up an actual subscription for a wine, auto-shipped?

**Speaker B (Eng Lead):** That's a much bigger thing. Subscriptions touch billing, inventory holds, cancellation flows. I'd keep that out of scope for v1. A reminder is a nudge; a subscription is a commitment. Let's not conflate them.

**Speaker A (PM):** Fair. Out of scope for now.

**Speaker C (Data):** One more wrinkle: a lot of what people buy is one-off — a gift, a wine for a specific dinner. If we remind someone to reorder the Champagne they bought for an anniversary, that's a bad experience. We'd want to only target wines that look like "everyday" repeat purchases, not occasion buys. We don't have a clean signal for that today, but we could approximate it.

**Speaker A (PM):** That's a good guardrail. Let's say we only nudge on wines under some price point and where the customer bought more than one bottle. We can refine.

**Speaker D (Lifecycle Marketing):** Who owns this, by the way? Is this a lifecycle campaign that my team runs in Braze, or is it a product feature engineering builds and maintains? Because that changes a lot about how we staff it.

**Speaker A (PM):** Honestly... I'm not sure yet. Let me take that one away. It might be a hybrid.

**Speaker B (Eng Lead):** And we should be clear about what success even looks like. Are we trying to drive second orders specifically? Total repeat revenue? I don't want to build this and then argue in three months about whether it worked.

**Speaker A (PM):** Yeah. I know we care about retention but I'll admit I don't have a target number in front of me. I'll work with Data to define the metric before we commit eng time.

**Speaker C (Data):** That'd help me too — what we measure changes how I'd set up the holdout group.

**Speaker A (PM):** Okay. So roughly: a Braze email that nudges repeat-friendly customers to reorder favorites, timed by a category heuristic to start, with frequency protection, excluding occasion buys, subscriptions out of scope. Open items: the success metric, who owns it, the frequency-cap prerequisite, and whether thirty days is even close. Let's reconvene once I've got the metric and ownership sorted.

**Speaker B (Eng Lead):** Works for me. Don't green-light eng until the frequency cap question is answered though — I mean it.

**Speaker A (PM):** Understood. Thanks all.

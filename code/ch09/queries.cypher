// queries.cypher
//
// The three "memory-augmented retrieval" queries from chapter 9, section 6.
// Each one closes a loop on a frustration from the chapter's cold open.


// tag::audit[]
// ── Query A: Audit ────────────────────────────────────────────────────────
//
// "Why did we recommend Carol for the Anthem engagement three weeks ago?"
//
// One hop from the Client into reasoning, made possible by the :TOUCHED edges
// the agent loop writes after each tool call. Without :TOUCHED this is a
// three-hop path through Message and the audit query is far less elegant.

MATCH (c:Client {name: $client_name})<-[:TOUCHED]-(step:ReasoningStep)
      <-[:HAS_STEP]-(trace:ReasoningTrace)
WHERE trace.completed_at >= datetime() - duration({weeks: $weeks_back})
OPTIONAL MATCH (step)-[:USES_TOOL]->(tc:ToolCall)
WITH trace, step, tc
ORDER BY trace.completed_at DESC, step.step_number ASC
RETURN trace.task                  AS task,
       trace.completed_at          AS when,
       trace.success               AS success,
       collect({
         step:    step.step_number,
         thought: step.thought,
         tool:    tc.tool_name,
         args:    tc.arguments,
         result:  tc.result
       })                          AS reasoning
ORDER BY when DESC;
// end::audit[]


// tag::bootstrap[]
// ── Query B: Personalized session bootstrap ───────────────────────────────
//
// At the start of every Sara session in the agent loop, fold her preferences
// for the relevant industry and her recent successful traces in that industry
// into the system prompt. All three regions touched in one query.

MATCH (u:User {identifier: $identifier})
OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)-[:APPLIES_TO]->(:Industry {name: $industry})
WITH u, collect(DISTINCT p.preference) AS preferences
CALL {
  WITH u
  MATCH (u)-[:HAS_CONVERSATION]->(:Conversation)-[:HAS_TRACE]->(rt:ReasoningTrace)
        -[:HAS_STEP]->(:ReasoningStep)-[:TOUCHED]->(:Client)
        -[:IN_INDUSTRY]->(:Industry {name: $industry})
  WHERE rt.success = true
    AND rt.completed_at >= datetime() - duration({days: 30})
  WITH DISTINCT rt
  ORDER BY rt.completed_at DESC
  LIMIT 5
  RETURN collect({task: rt.task, outcome: rt.outcome}) AS recent_wins
}
RETURN preferences, recent_wins;
// end::bootstrap[]


// tag::case_based[]
// ── Query C: Case-based reasoning over reasoning memory ───────────────────
//
// "Sara wants a senior team for a healthcare migration. What did we decide
//  the last few times we faced a similar task, and did it work?"
//
// Vector search over the task_embedding the library populates on every
// ReasoningTrace. The result is a small set of analogous past reasoning
// trajectories the agent can pre-load into context.

WITH genai.vector.encode($task_query, "OpenAI", {token: $openai_key}) AS q
CALL db.index.vector.queryNodes("trace_task_embedding", 5, q) YIELD node AS rt, score
WHERE rt.success = true
MATCH (rt)-[:HAS_STEP]->(step:ReasoningStep)-[:USES_TOOL]->(tc:ToolCall)
WITH rt, score, step, tc
ORDER BY rt.task, step.step_number ASC
WITH rt, score,
     collect({step: step.step_number, tool: tc.tool_name, args: tc.arguments})[..10] AS tool_path
RETURN rt.task                     AS task,
       rt.outcome                  AS outcome,
       score                       AS similarity,
       tool_path
ORDER BY similarity DESC;
// end::case_based[]


// ── Query C, library-wrapper form ─────────────────────────────────────────
//
// Equivalent to the raw cypher above, but using the library's
// reasoning.search_traces helper. Prefer this form in application code; the
// raw form above is what the wrapper compiles to.
//
// (This block is run from Python, not Cypher; included here as documentation.)
//
// async with MemoryClient(settings) as client:
//     results = await client.reasoning.search_traces(
//         query="Staff a senior team for a healthcare migration",
//         limit=5,
//         success_only=True,
//     )

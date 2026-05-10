/*
 * Copyright EDDI contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.labs.eddi.modules.llm.impl;

import ai.labs.eddi.configs.agents.IRestAgentStore;
import ai.labs.eddi.configs.rag.model.RagConfiguration;
import ai.labs.eddi.configs.workflows.IRestWorkflowStore;
import ai.labs.eddi.engine.memory.IConversationMemory;
import ai.labs.eddi.engine.memory.IDataFactory;
import ai.labs.eddi.engine.runtime.client.configuration.IResourceClientLibrary;
import ai.labs.eddi.modules.llm.model.LlmConfiguration;
import ai.labs.eddi.modules.llm.model.LlmConfiguration.KnowledgeBaseReference;
import ai.labs.eddi.modules.llm.model.LlmConfiguration.RagDefaults;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingStore;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Discovers RAG configurations from the workflow, performs retrieval, and
 * formats context for injection into LLM system messages.
 *
 * <p>
 * Follows the same WorkflowTraversal pattern as httpcall and mcpcalls discovery
 * in {@link AgentOrchestrator}.
 * </p>
 */
@ApplicationScoped
public class RagContextProvider {

    private static final Logger LOGGER = Logger.getLogger(RagContextProvider.class);
    private static final String RAG_TYPE = "eddi://ai.labs.rag";

    private final IRestAgentStore restAgentStore;
    private final IRestWorkflowStore restWorkflowStore;
    private final IResourceClientLibrary resourceClientLibrary;
    private final EmbeddingModelFactory embeddingModelFactory;
    private final EmbeddingStoreFactory embeddingStoreFactory;
    private final IDataFactory dataFactory;

    @Inject
    public RagContextProvider(IRestAgentStore restAgentStore, IRestWorkflowStore restWorkflowStore,
            IResourceClientLibrary resourceClientLibrary,
            EmbeddingModelFactory embeddingModelFactory, EmbeddingStoreFactory embeddingStoreFactory,
            IDataFactory dataFactory) {
        this.restAgentStore = restAgentStore;
        this.restWorkflowStore = restWorkflowStore;
        this.resourceClientLibrary = resourceClientLibrary;
        this.embeddingModelFactory = embeddingModelFactory;
        this.embeddingStoreFactory = embeddingStoreFactory;
        this.dataFactory = dataFactory;
    }

    public Optional<String> retrieveContext(IConversationMemory memory, LlmConfiguration.Task task, String userQuery) {
        List<KnowledgeBaseReference> kbRefs = task.getKnowledgeBases();
        boolean hasExplicitRefs = kbRefs != null && !kbRefs.isEmpty();
        boolean useWorkflowDiscovery = !hasExplicitRefs && Boolean.TRUE.equals(task.getEnableWorkflowRag());

        if (!hasExplicitRefs && !useWorkflowDiscovery) {
            return Optional.empty();
        }

        var ragSteps = WorkflowTraversal.discoverConfigs(memory, RAG_TYPE, RagConfiguration.class, restAgentStore,
                restWorkflowStore,
                resourceClientLibrary);

        if (ragSteps.isEmpty()) {
            LOGGER.debug("No RAG steps found in workflow");
            return Optional.empty();
        }

        List<RetrievalResult> allResults = new ArrayList<>();
        List<Map<String, Object>> traceEntries = new ArrayList<>();
        var currentStep = memory.getCurrentStep();
        String taskId = task.getId() != null ? task.getId() : "default";

        for (var step : ragSteps) {
            RagConfiguration ragConfig = step.config();
            String kbName = ragConfig.getName();

            if (!shouldUseKb(kbRefs, kbName, useWorkflowDiscovery)) {
                continue;
            }

            resolveRetrievalParams(ragConfig, kbRefs, kbName, useWorkflowDiscovery, task)
                    .ifPresent(params -> retrieveFromKb(ragConfig, kbName, params, userQuery, allResults, traceEntries));
        }

        storeTraceInMemory(currentStep, taskId, traceEntries);

        if (allResults.isEmpty()) {
            return Optional.empty();
        }

        String formattedContext = formatRagContext(allResults);
        var ragContextData = dataFactory.createData("rag:context:" + taskId, formattedContext);
        currentStep.storeData(ragContextData);

        return Optional.of(formattedContext);
    }

    private boolean shouldUseKb(List<KnowledgeBaseReference> kbRefs, String kbName, boolean useWorkflowDiscovery) {
        return useWorkflowDiscovery || (kbRefs != null && kbRefs.stream().anyMatch(r -> kbName.equals(r.getName())));
    }

    private Optional<RetrievalParams> resolveRetrievalParams(RagConfiguration ragConfig, List<KnowledgeBaseReference> kbRefs,
                                                             String kbName, boolean useWorkflowDiscovery,
                                                             LlmConfiguration.Task task) {
        if (useWorkflowDiscovery) {
            RagDefaults defaults = task.getRagDefaults();
            int maxResults = defaults != null && defaults.getMaxResults() != null
                    ? defaults.getMaxResults()
                    : ragConfig.getMaxResults();
            double minScore = defaults != null && defaults.getMinScore() != null
                    ? defaults.getMinScore()
                    : ragConfig.getMinScore();
            return Optional.of(new RetrievalParams(maxResults, minScore));
        }

        if (kbRefs == null) {
            return Optional.empty();
        }

        return kbRefs.stream()
                .filter(r -> kbName.equals(r.getName()))
                .findFirst()
                .map(ref -> new RetrievalParams(
                        ref.getMaxResults() != null ? ref.getMaxResults() : ragConfig.getMaxResults(),
                        ref.getMinScore() != null ? ref.getMinScore() : ragConfig.getMinScore()));
    }

    private void retrieveFromKb(RagConfiguration ragConfig, String kbName, RetrievalParams params,
                                String userQuery, List<RetrievalResult> allResults,
                                List<Map<String, Object>> traceEntries) {
        try {
            EmbeddingModel embeddingModel = embeddingModelFactory.getOrCreate(ragConfig);
            EmbeddingStore<TextSegment> store = embeddingStoreFactory.getOrCreate(ragConfig, kbName);

            ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(store).embeddingModel(embeddingModel)
                    .maxResults(params.maxResults()).minScore(params.minScore()).build();

            List<Content> relevant = retriever.retrieve(Query.from(userQuery));

            LOGGER.infof("RAG retrieval from KB '%s': %d results (maxResults=%d, minScore=%.2f)",
                    kbName, relevant.size(), params.maxResults(), params.minScore());

            traceEntries.add(buildSuccessTrace(ragConfig, kbName, params, relevant.size()));
            allResults.addAll(relevant.stream().map(c -> new RetrievalResult(kbName, c)).toList());

        } catch (Exception e) {
            LOGGER.warnf(e, "Failed to retrieve from KB '%s': %s", kbName, e.getMessage());
            traceEntries.add(buildErrorTrace(kbName, e));
        }
    }

    private Map<String, Object> buildSuccessTrace(RagConfiguration ragConfig, String kbName,
                                                  RetrievalParams params, int retrievedCount) {
        Map<String, Object> trace = new HashMap<>();
        trace.put("kb", kbName);
        trace.put("provider", ragConfig.getEmbeddingProvider());
        trace.put("storeType", ragConfig.getStoreType());
        trace.put("maxResults", params.maxResults());
        trace.put("minScore", params.minScore());
        trace.put("retrievedCount", retrievedCount);
        return trace;
    }

    private Map<String, Object> buildErrorTrace(String kbName, Exception e) {
        Map<String, Object> trace = new HashMap<>();
        trace.put("kb", kbName);
        trace.put("error", e.getMessage());
        return trace;
    }

    private void storeTraceInMemory(IConversationMemory.IWritableConversationStep currentStep,
                                    String taskId, List<Map<String, Object>> traceEntries) {
        if (traceEntries.isEmpty()) {
            return;
        }

        var ragTraceData = dataFactory.createData("rag:trace:" + taskId, traceEntries);
        currentStep.storeData(ragTraceData);
    }

    /**
     * Formats retrieval results into a structured context string for the LLM.
     */
    private String formatRagContext(List<RetrievalResult> results) {
        StringBuilder sb = new StringBuilder();
        String currentKb = null;

        for (RetrievalResult result : results) {
            if (!result.kbName().equals(currentKb)) {
                if (currentKb != null) {
                    sb.append("\n");
                }
                sb.append("### Source: ").append(result.kbName()).append("\n\n");
                currentKb = result.kbName();
            }
            sb.append(result.content().textSegment() != null && result.content().textSegment().text() != null
                    ? result.content().textSegment().text()
                    : "").append("\n\n");
        }

        return sb.toString().trim();
    }

    record RetrievalResult(String kbName, Content content) {
    }

    record RetrievalParams(int maxResults, double minScore) {
    }

}

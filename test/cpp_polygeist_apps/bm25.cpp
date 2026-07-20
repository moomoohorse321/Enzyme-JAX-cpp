template <enzyme::size_t Position, enzyme::size_t NumTokens>
__attribute__((always_inline)) static inline float
term_frequency(const enzyme::tensor<int32_t, NumTokens> &document,
               int32_t term) {
  if constexpr (Position == NumTokens) {
    return 0.0f;
  } else {
    float present = document[Position] == term ? 1.0f : 0.0f;
    return present + term_frequency<Position + 1>(document, term);
  }
}

template <enzyme::size_t Document, enzyme::size_t NumDocuments,
          enzyme::size_t NumTokens>
__attribute__((always_inline)) static inline float document_frequency(
    const enzyme::tensor<int32_t, NumDocuments, NumTokens> &documents,
    int32_t term) {
  if constexpr (Document == NumDocuments) {
    return 0.0f;
  } else {
    float frequency = term_frequency<0>(documents[Document], term);
    float present = frequency > 0.0f ? 1.0f : 0.0f;
    return present + document_frequency<Document + 1>(documents, term);
  }
}

template <enzyme::size_t Document, enzyme::size_t NumDocuments>
__attribute__((always_inline)) static inline float total_document_length(
    const enzyme::tensor<int32_t, NumDocuments> &document_lengths) {
  if constexpr (Document == NumDocuments) {
    return 0.0f;
  } else {
    return static_cast<float>(document_lengths[Document]) +
           total_document_length<Document + 1>(document_lengths);
  }
}

template <enzyme::size_t QueryIndex, enzyme::size_t NumDocuments,
          enzyme::size_t NumTokens, enzyme::size_t NumQueryTerms>
__attribute__((always_inline)) static inline float score_document(
    enzyme::size_t document_index,
    const enzyme::tensor<int32_t, NumDocuments, NumTokens> &documents,
    const enzyme::tensor<int32_t, NumDocuments> &document_lengths,
    const enzyme::tensor<int32_t, NumQueryTerms> &query,
    float average_document_length) {
  if constexpr (QueryIndex == NumQueryTerms) {
    return 0.0f;
  } else {
    constexpr float k1 = 1.2f;
    constexpr float b = 0.75f;
    int32_t term = query[QueryIndex];
    float tf = term_frequency<0>(documents[document_index], term);
    float df = document_frequency<0>(documents, term);
    float idf = __builtin_logf(
        (static_cast<float>(NumDocuments) - df + 0.5f) / (df + 0.5f) +
        1.0f);
    float length_ratio =
        static_cast<float>(document_lengths[document_index]) /
        average_document_length;
    float denominator = tf + k1 * (1.0f - b + b * length_ratio);
    float contribution = idf * (tf * (k1 + 1.0f)) / denominator;
    return contribution +
           score_document<QueryIndex + 1>(
               document_index, documents, document_lengths, query,
               average_document_length);
  }
}

template <enzyme::size_t NumDocuments, enzyme::size_t NumTokens,
          enzyme::size_t NumQueryTerms>
void bm25(enzyme::tensor<float, NumDocuments> &scores,
          const enzyme::tensor<int32_t, NumDocuments, NumTokens> &documents,
          const enzyme::tensor<int32_t, NumDocuments> &document_lengths,
          const enzyme::tensor<int32_t, NumQueryTerms> &query) {
  float average_document_length =
      total_document_length<0>(document_lengths) /
      static_cast<float>(NumDocuments);
  for (enzyme::size_t document = 0; document < NumDocuments; ++document)
    scores[document] = score_document<0>(
        document, documents, document_lengths, query,
        average_document_length);
}

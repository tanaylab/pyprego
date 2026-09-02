#!/usr/bin/env Rscript
# Generate golden master test data from R prego package for Python comparison.
# Outputs JSON files in the same directory as this script.

library(prego)
library(jsonlite)

# Determine output directory: use the directory of this script
args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("--file=", args, value = TRUE)
if (length(script_arg) > 0) {
    out_dir <- dirname(normalizePath(sub("--file=", "", script_arg), mustWork = FALSE))
} else {
    out_dir <- getwd()
}
cat("Output directory:", out_dir, "\n")

write_json_file <- function(data, filename) {
    path <- file.path(out_dir, filename)
    write(toJSON(data, auto_unbox = TRUE, digits = 15, na = "null"), path)
    cat("Wrote:", path, "\n")
}

# ============================================================================
# (a) Save example data (subsets for manageable size)
# ============================================================================
cat("\n=== Saving example data ===\n")

# Use first 100 sequences and their responses for manageable test sizes
n_seq <- 100
seq_subset <- sequences_example[1:n_seq]
resp_subset <- response_mat_example[1:n_seq, , drop = FALSE]

# Also subset cluster data
n_clust <- 200
clust_seq_subset <- cluster_sequences_example[1:n_clust]
clust_mat_subset <- cluster_mat_example[1:n_clust, , drop = FALSE]
clust_subset <- clusters_example[1:n_clust]

write_json_file(list(
    sequences = seq_subset,
    response_mat = as.data.frame(resp_subset),
    cluster_sequences = clust_seq_subset,
    cluster_mat = as.data.frame(clust_mat_subset),
    clusters = clust_subset,
    n_seq = n_seq,
    n_clust = n_clust,
    seq_length = nchar(seq_subset[1])
), "example_data.json")

# ============================================================================
# (b) compute_pwm: create a simple PSSM from known values, score sequences
# ============================================================================
cat("\n=== compute_pwm ===\n")

# Create a small test PSSM (8 positions, representing GATA motif-like)
test_pssm <- data.frame(
    pos = 0:7,
    A = c(0.1, 0.05, 0.8, 0.05, 0.8, 0.1, 0.1, 0.2),
    C = c(0.2, 0.05, 0.05, 0.05, 0.05, 0.3, 0.2, 0.3),
    G = c(0.6, 0.85, 0.1, 0.05, 0.1, 0.5, 0.6, 0.3),
    T = c(0.1, 0.05, 0.05, 0.85, 0.05, 0.1, 0.1, 0.2)
)

# Score with default params (bidirect=TRUE, prior=0.01, logSumExp)
tryCatch({
    scores_default <- compute_pwm(seq_subset, test_pssm, bidirect = TRUE, prior = 0.01, func = "logSumExp")
    scores_max <- compute_pwm(seq_subset, test_pssm, bidirect = TRUE, prior = 0.01, func = "max")
    scores_norc <- compute_pwm(seq_subset, test_pssm, bidirect = FALSE, prior = 0.01, func = "logSumExp")

    write_json_file(list(
        pssm = as.data.frame(test_pssm),
        scores_default = as.numeric(scores_default),
        scores_max = as.numeric(scores_max),
        scores_norc = as.numeric(scores_norc),
        prior = 0.01,
        bidirect = TRUE
    ), "compute_pwm.json")
    cat("compute_pwm: SUCCESS\n")
}, error = function(e) {
    cat("compute_pwm: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (c) screen_kmers
# ============================================================================
cat("\n=== screen_kmers ===\n")

tryCatch({
    kmers_result <- screen_kmers(seq_subset, resp_subset, kmer_length = 5, min_cor = 0.05, seed = 42)

    write_json_file(list(
        kmer_length = 5,
        min_cor = 0.05,
        seed = 42,
        result = as.data.frame(kmers_result)
    ), "screen_kmers.json")
    cat("screen_kmers: SUCCESS -", nrow(kmers_result), "kmers found\n")
}, error = function(e) {
    cat("screen_kmers: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (d) generate_kmers
# ============================================================================
cat("\n=== generate_kmers ===\n")

tryCatch({
    kmers_5 <- generate_kmers(5)

    write_json_file(list(
        k = 5,
        kmers = kmers_5,
        n_kmers = length(kmers_5)
    ), "generate_kmers.json")
    cat("generate_kmers: SUCCESS -", length(kmers_5), "kmers\n")
}, error = function(e) {
    cat("generate_kmers: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (e) kmer_matrix
# ============================================================================
cat("\n=== kmer_matrix ===\n")

tryCatch({
    # R kmer_matrix takes kmer_length (integer), not a list of kmers
    km <- kmer_matrix(seq_subset[1:20], kmer_length = 4)

    # Save only the first 20 columns to keep file size manageable
    km_subset <- km[, 1:min(20, ncol(km)), drop = FALSE]

    write_json_file(list(
        sequences = seq_subset[1:20],
        kmer_length = 4,
        colnames = colnames(km),
        matrix_subset = as.data.frame(km_subset),
        matrix_subset_colnames = colnames(km_subset),
        full_dim = c(nrow(km), ncol(km))
    ), "kmer_matrix.json")
    cat("kmer_matrix: SUCCESS -", nrow(km), "x", ncol(km), "\n")
}, error = function(e) {
    cat("kmer_matrix: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (f) pssm_cor
# ============================================================================
cat("\n=== pssm_cor ===\n")

tryCatch({
    pssm1 <- data.frame(
        pos = 0:5,
        A = c(0.9, 0.05, 0.05, 0.8, 0.1, 0.2),
        C = c(0.03, 0.85, 0.05, 0.1, 0.3, 0.3),
        G = c(0.04, 0.05, 0.85, 0.05, 0.5, 0.3),
        T = c(0.03, 0.05, 0.05, 0.05, 0.1, 0.2)
    )
    pssm2 <- data.frame(
        pos = 0:3,
        A = c(0.85, 0.1, 0.1, 0.7),
        C = c(0.05, 0.8, 0.1, 0.1),
        G = c(0.05, 0.05, 0.7, 0.1),
        T = c(0.05, 0.05, 0.1, 0.1)
    )

    cor_spearman <- pssm_cor(pssm1, pssm2, method = "spearman", prior = 0.01)
    cor_pearson <- pssm_cor(pssm1, pssm2, method = "pearson", prior = 0.01)

    write_json_file(list(
        pssm1 = as.data.frame(pssm1),
        pssm2 = as.data.frame(pssm2),
        cor_spearman = cor_spearman,
        cor_pearson = cor_pearson,
        prior = 0.01
    ), "pssm_cor.json")
    cat("pssm_cor: SUCCESS - spearman:", cor_spearman, "pearson:", cor_pearson, "\n")
}, error = function(e) {
    cat("pssm_cor: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (g) pssm_trim
# ============================================================================
cat("\n=== pssm_trim ===\n")

tryCatch({
    # PSSM with low-info edges
    pssm_with_edges <- data.frame(
        pos = 0:9,
        A = c(0.25, 0.25, 0.9, 0.05, 0.85, 0.1, 0.8, 0.25, 0.25, 0.25),
        C = c(0.25, 0.25, 0.03, 0.85, 0.05, 0.3, 0.1, 0.25, 0.25, 0.25),
        G = c(0.25, 0.25, 0.04, 0.05, 0.05, 0.5, 0.05, 0.25, 0.25, 0.25),
        T = c(0.25, 0.25, 0.03, 0.05, 0.05, 0.1, 0.05, 0.25, 0.25, 0.25)
    )

    trimmed <- pssm_trim(pssm_with_edges, bits_thresh = 0.1)

    write_json_file(list(
        input_pssm = as.data.frame(pssm_with_edges),
        bits_thresh = 0.1,
        trimmed_pssm = as.data.frame(trimmed),
        trimmed_nrow = nrow(trimmed)
    ), "pssm_trim.json")
    cat("pssm_trim: SUCCESS -", nrow(pssm_with_edges), "->", nrow(trimmed), "positions\n")
}, error = function(e) {
    cat("pssm_trim: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (h) bits_per_pos
# ============================================================================
cat("\n=== bits_per_pos ===\n")

tryCatch({
    bits <- bits_per_pos(test_pssm, prior = 0.01)

    write_json_file(list(
        pssm = as.data.frame(test_pssm),
        prior = 0.01,
        bits = as.numeric(bits)
    ), "bits_per_pos.json")
    cat("bits_per_pos: SUCCESS -", length(bits), "values\n")
}, error = function(e) {
    cat("bits_per_pos: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (i) consensus_from_pssm
# ============================================================================
cat("\n=== consensus_from_pssm ===\n")

tryCatch({
    # R consensus_from_pssm uses different default thresholds: single_thresh=0.4, double_thresh=0.6
    consensus <- consensus_from_pssm(test_pssm, single_thresh = 0.4, double_thresh = 0.6)

    write_json_file(list(
        pssm = as.data.frame(test_pssm),
        single_thresh = 0.4,
        double_thresh = 0.6,
        consensus = consensus
    ), "consensus_from_pssm.json")
    cat("consensus_from_pssm: SUCCESS -", consensus, "\n")
}, error = function(e) {
    cat("consensus_from_pssm: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (j) rc (reverse complement)
# ============================================================================
cat("\n=== rc ===\n")

tryCatch({
    test_seqs <- c("ACGTACGT", "AAACCCTTTGGG", "GATCGATC", "AAAAAA", "TTTCCCGGG")
    rc_results <- rc(test_seqs)

    write_json_file(list(
        inputs = test_seqs,
        outputs = rc_results
    ), "rc.json")
    cat("rc: SUCCESS\n")
}, error = function(e) {
    cat("rc: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (k) calc_sequences_dinucs
# ============================================================================
cat("\n=== calc_sequences_dinucs ===\n")

tryCatch({
    dinuc_mat <- calc_sequences_dinucs(seq_subset[1:20])

    write_json_file(list(
        sequences = seq_subset[1:20],
        matrix = as.data.frame(dinuc_mat),
        colnames = colnames(dinuc_mat)
    ), "calc_sequences_dinucs.json")
    cat("calc_sequences_dinucs: SUCCESS -", nrow(dinuc_mat), "x", ncol(dinuc_mat), "\n")
}, error = function(e) {
    cat("calc_sequences_dinucs: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# Additional useful golden master data
# ============================================================================

# pssm_theoretical_max/min
cat("\n=== pssm_theoretical_max/min ===\n")

tryCatch({
    theo_max <- pssm_theoretical_max(test_pssm, prior = 0.01, regularization = 0.01)
    theo_min <- pssm_theoretical_min(test_pssm, prior = 0.01, regularization = 0.01)
    theo_q50 <- pssm_quantile(test_pssm, 0.5, prior = 0.01, regularization = 0.01)
    theo_q85 <- pssm_quantile(test_pssm, 0.85, prior = 0.01, regularization = 0.01)

    write_json_file(list(
        pssm = as.data.frame(test_pssm),
        prior = 0.01,
        regularization = 0.01,
        theoretical_max = theo_max,
        theoretical_min = theo_min,
        quantile_50 = theo_q50,
        quantile_85 = theo_q85
    ), "pssm_theoretical.json")
    cat("pssm_theoretical: SUCCESS - max:", theo_max, "min:", theo_min, "\n")
}, error = function(e) {
    cat("pssm_theoretical: FAILED -", conditionMessage(e), "\n")
})

# pssm_rc
cat("\n=== pssm_rc ===\n")

tryCatch({
    rc_pssm <- pssm_rc(test_pssm)

    write_json_file(list(
        input_pssm = as.data.frame(test_pssm),
        rc_pssm = as.data.frame(rc_pssm)
    ), "pssm_rc.json")
    cat("pssm_rc: SUCCESS\n")
}, error = function(e) {
    cat("pssm_rc: FAILED -", conditionMessage(e), "\n")
})

# pssm_diff (KL divergence)
cat("\n=== pssm_diff ===\n")

tryCatch({
    pssm1 <- data.frame(
        pos = 0:5,
        A = c(0.9, 0.05, 0.05, 0.8, 0.1, 0.2),
        C = c(0.03, 0.85, 0.05, 0.1, 0.3, 0.3),
        G = c(0.04, 0.05, 0.85, 0.05, 0.5, 0.3),
        T = c(0.03, 0.05, 0.05, 0.05, 0.1, 0.2)
    )
    pssm2 <- data.frame(
        pos = 0:3,
        A = c(0.85, 0.1, 0.1, 0.7),
        C = c(0.05, 0.8, 0.1, 0.1),
        G = c(0.05, 0.05, 0.7, 0.1),
        T = c(0.05, 0.05, 0.1, 0.1)
    )

    kl_div <- pssm_diff(pssm1, pssm2, prior = 0.01)

    write_json_file(list(
        pssm1 = as.data.frame(pssm1),
        pssm2 = as.data.frame(pssm2),
        kl_divergence = kl_div,
        prior = 0.01
    ), "pssm_diff.json")
    cat("pssm_diff: SUCCESS - kl:", kl_div, "\n")
}, error = function(e) {
    cat("pssm_diff: FAILED -", conditionMessage(e), "\n")
})

# ============================================================================
# (r) calc_freq_local_pwm: expected local PWM scores over base frequencies
# ============================================================================
cat("\n=== calc_freq_local_pwm ===\n")

if (!exists("calc_freq_local_pwm")) {
    cat("calc_freq_local_pwm: SKIPPED - not available in this prego version\n")
} else {
    tryCatch({
        # Two motifs of different lengths, so padding past the shorter motif
        # and the per-motif NA tail are exercised.
        m4 <- data.frame(
            motif = "M4", pos = 1:4,
            A = c(0.7, 0.1, 0.1, 0.25), C = c(0.1, 0.7, 0.1, 0.25),
            G = c(0.1, 0.1, 0.7, 0.25), T = c(0.1, 0.1, 0.1, 0.25)
        )
        m6 <- data.frame(
            motif = "M6", pos = 1:6,
            A = c(0.9, 0.05, 0.4, 0.25, 0.1, 0.6), C = c(0.05, 0.9, 0.2, 0.25, 0.1, 0.2),
            G = c(0.03, 0.03, 0.2, 0.25, 0.7, 0.1), T = c(0.02, 0.02, 0.2, 0.25, 0.1, 0.1)
        )
        small_db <- rbind(m4, m6)
        small_mdb <- create_motif_db(small_db)

        # A wider database with motifs of many different lengths, to exercise
        # the length-sorted blocking on a realistic mixture.
        all_db <- all_motif_datasets()
        lens <- tapply(all_db$pos, all_db$motif, length)
        picked <- unlist(lapply(sort(unique(lens)), function(L) names(lens)[lens == L][1]))
        picked <- picked[seq(1, length(picked), length.out = min(15, length(picked)))]
        wide_db <- all_db[all_db$motif %in% picked, c("motif", "pos", "A", "C", "G", "T")]
        wide_mdb <- create_motif_db(wide_db)

        set.seed(60427)
        n_pos <- 64
        q_rand <- matrix(stats::runif(n_pos * 4, 0.05, 1), nrow = n_pos, ncol = 4)
        q_rand <- q_rand / rowSums(q_rand)

        seq_onehot <- "ACGTACGTTGCAAGGTCCATACGTACGTTGCAAGGTCCAT"
        q_onehot <- diag(4)[match(strsplit(seq_onehot, "")[[1]], c("A", "C", "G", "T")), ]
        q_flat <- matrix(0.25, nrow = 40, ncol = 4)

        score_all <- function(q, mdb) {
            out <- list()
            for (combine in c("multiply", "sum")) {
                for (bidirect in c(TRUE, FALSE)) {
                    key <- paste0(combine, "_", if (bidirect) "bidirect" else "forward")
                    out[[key]] <- as.data.frame(calc_freq_local_pwm(
                        q, mdb, combine = combine, bidirect = bidirect
                    ))
                }
            }
            out
        }

        # compute_local_pwm on the same one-hot sequence, for the anchor that
        # a certain ensemble scores exactly like the sequence it encodes.
        small_tidy <- motif_db_to_dataframe(small_mdb)
        local_ref <- list()
        for (bidirect in c(TRUE, FALSE)) {
            key <- if (bidirect) "bidirect" else "forward"
            local_ref[[key]] <- as.data.frame(t(sapply(
                colnames(small_mdb@mat),
                function(mo) as.numeric(compute_local_pwm(
                    seq_onehot, small_tidy[small_tidy$motif == mo, c("A", "C", "G", "T")],
                    bidirect = bidirect, prior = small_mdb@prior
                ))
            )))
        }

        write_json_file(list(
            prior = small_mdb@prior,
            small_db = as.data.frame(small_db),
            small_motifs = colnames(small_mdb@mat),
            small_lengths = as.integer(small_mdb@motif_lengths),
            wide_db = as.data.frame(wide_db),
            wide_motifs = colnames(wide_mdb@mat),
            wide_lengths = as.integer(wide_mdb@motif_lengths),
            q_random = as.data.frame(q_rand),
            q_onehot = as.data.frame(q_onehot),
            q_flat_positions = nrow(q_flat),
            onehot_sequence = seq_onehot,
            small_random = score_all(q_rand, small_mdb),
            small_onehot = score_all(q_onehot, small_mdb),
            small_flat = score_all(q_flat, small_mdb),
            wide_random = score_all(q_rand, wide_mdb),
            compute_local_pwm_onehot = local_ref
        ), "calc_freq_local_pwm.json")
        cat("calc_freq_local_pwm: SUCCESS -", length(picked), "wide motifs\n")
    }, error = function(e) {
        cat("calc_freq_local_pwm: FAILED -", conditionMessage(e), "\n")
    })
}

cat("\n=== Done ===\n")

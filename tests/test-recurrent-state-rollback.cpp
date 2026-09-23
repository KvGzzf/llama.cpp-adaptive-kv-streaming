#include "arg.h"
#include "common.h"
#include "llama.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent-spill.h"

#include <algorithm>
#include <clocale>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx_src = make_ctx(params, model);
    llama_context * ctx_dst = make_ctx(params, model);
    llama_context * ctx_stock = nullptr;
    if (const char * spill_env = std::getenv("LLAMA_RS_HOST_SPILL")) {
        const std::string original(spill_env);
#ifdef _WIN32
        _putenv_s("LLAMA_RS_HOST_SPILL", "");
#else
        unsetenv("LLAMA_RS_HOST_SPILL");
#endif
        ctx_stock = make_ctx(params, model);
#ifdef _WIN32
        _putenv_s("LLAMA_RS_HOST_SPILL", original.c_str());
#else
        setenv("LLAMA_RS_HOST_SPILL", original.c_str(), 1);
#endif
        if (!ctx_stock) return 1;
    }
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (std::getenv("LLAMA_RS_HOST_SPILL")) {
        auto * hybrid=static_cast<llama_memory_hybrid *>(llama_get_memory(ctx_src));
        auto * recurrent=hybrid->get_mem_recr();
        for (size_t layer=0;layer<recurrent->r_l.size();++layer) {
            for (auto * tensor : {recurrent->r_l[layer],recurrent->s_l[layer]}) {
                if (tensor && tensor->ne[1] != recurrent->size) {
                    fprintf(stderr,"%s : recurrent device storage still contains rollback planes\n",__func__);
                    return 1;
                }
            }
        }
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (ctx_stock && !decode_tokens(ctx_stock, tokens, n_tokens)) {
        fprintf(stderr,"%s : stock reference decode failed\n",__func__);
        return 1;
    }
    if (ctx_stock) {
        const float * spilled=llama_get_logits(ctx_src);
        const float * stock=llama_get_logits(ctx_stock);
        if (!spilled || !stock) return 1;
        for (int token=0;token<n_vocab;++token) {
            if (std::fabs(spilled[token]-stock[token]) > 1e-5f) {
                fprintf(stderr,"%s : initial spilled/stock logit mismatch at token %d\n",__func__,token);
                return 1;
            }
        }
    }


    if (std::getenv("LLAMA_RS_HOST_SPILL")) {
        auto * hybrid=static_cast<llama_memory_hybrid *>(llama_get_memory(ctx_src));
        auto * recurrent=hybrid->get_mem_recr();
        if (!recurrent->complete_spill() || !recurrent->spill_bank()) {
            fprintf(stderr,"%s : host spill did not publish\n",__func__);
            return 1;
        }
        const auto generation=recurrent->spill_generation();
        for (size_t layer=0;layer<recurrent->r_l.size();++layer) for (bool value : {false,true}) {
            auto * tensor=value ? recurrent->s_l[layer] : recurrent->r_l[layer];
            if (!tensor) continue;
            const size_t row=ggml_row_size(tensor->type,tensor->ne[0]);
            std::vector<uint8_t> stock(row);
            for (uint32_t slot=1;slot<=n_rs_seq;++slot) {
                const auto * captured=recurrent->spill_bank()->snapshot_data(layer,value,slot,generation);
                if (!captured) {
                    fprintf(stderr,"%s : missing host snapshot layer %zu slot %u\n",__func__,layer,slot);
                    return 1;
                }
                auto * stock_cache=static_cast<llama_memory_hybrid *>(llama_get_memory(ctx_stock))->get_mem_recr();
                ggml_backend_tensor_get(value ? stock_cache->s_l[layer] : stock_cache->r_l[layer],stock.data(),slot*recurrent->size*row,row);
                if (std::memcmp(stock.data(),captured,row) != 0) {
                    fprintf(stderr,"%s : host snapshot differs from stock layer %zu slot %u\n",__func__,layer,slot);
                    return 1;
                }
            }
        }
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            for (int token = 0; token < n_vocab; ++token) {
                if (std::fabs(logits_src[token] - logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial")) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (std::fabs(logits_src_replay[i][token] - logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);
    if (ctx_stock) llama_free(ctx_stock);
    return 0;
}

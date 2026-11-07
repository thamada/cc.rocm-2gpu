/**
 * 2 GPU の転送帯域ベンチマーク
 *
 * 起動:
 *   make run
 *   make run GPU_A=0 GPU_B=1 MAX_MB=16384 ITERS=0
 *   ./p2p-bw [gpu_a] [gpu_b] [max_mb] [iters]
 *   P2P_CHUNK_MB=8 ./p2p-bw          # ホスト経由転送のチャンクサイズ (1..256)
 *
 * iters が 0 のときは、転送サイズごとに反復回数を自動で決める。
 *
 * ---------------------------------------------------------------------------
 * 何を測るか
 * ---------------------------------------------------------------------------
 * このプログラムは次の 3 種類を、同じバッファサイズの列（4 KiB から最大サイズ
 * まで 4 倍ずつ）で測る。
 *
 *  1. GPU 間の単方向
 *       GPU A の VRAM → GPU B の VRAM
 *       GPU B の VRAM → GPU A の VRAM
 *  2. GPU 間の双方向
 *       上の 2 本を同時に流したときの合計帯域
 *  3. Host ↔ VRAM
 *       書き込み (Host → VRAM) と読み出し (VRAM → Host) を別々に測る。
 *       各方向で「その GPU だけ」と「2 GPU が同時」の両方を出す。
 *
 * 帯域の単位は GB/s = 10^9 バイト/秒（GiB/s = 2^30 ではない）。
 * 表示の「1回転送」は 1 回分の所要時間。双方向と「同時合計」は、
 * 同じ時間に動いた両方向（または両 GPU）のバイト数を足して割った値。
 *
 * サマリの理論ピークは、GPU からルートポートまでの各ホップで sysfs が示す
 * 交渉済み速度・幅のうち、片方向ペイロードが最も小さいリンクである。
 * デバイス直結が 32 GT/s x16 でも、上流のブリッジが 8 GT/s x16 なら
 * 後者が Host 転送の上限になる。端点の速度だけで世代を決めない。
 * 符号化は Gen1/2 が 8b/10b、Gen3–5 が 128b/130b、Gen6 が FLIT の 242/256。
 * 実行効率は 実測 ÷ その最遅リンクの片方向ピーク。
 *   Host の単独     その GPU の経路で最も遅いホップ
 *   Host の同時合計 両 GPU それぞれの最遅ホップの和
 *   GPU 間の片方向  両 GPU の最遅ホップのうち遅い方
 *   GPU 間の双方向  上の 2 倍（PCIe は全二重）
 * xGMI のとき GPU 間の帯域上限は PCIe では置けないので、その行は
 * 「理論ピーク不明」にする。Host↔VRAM は xGMI でも各 GPU の PCIe 経路を使う。
 *
 * ---------------------------------------------------------------------------
 * GPU 間の経路
 * ---------------------------------------------------------------------------
 * hipDeviceCanAccessPeer が両方向で真なら、hipMemcpyPeerAsync で VRAM から
 * VRAM へ直接コピーする。xGMI や PCIe ピアがその経路になる。
 *
 * ピアアクセスが無い（このマシンの Radeon 同士はこちら）場合は、ドライバに
 * 直列の中継を任せない。ピン留めしたホストメモリを 2 枚用意し、
 * 「GPU → ホスト」と「ホスト → もう一方の GPU」をチャンク単位で重ねる。
 * 1 枚のバッファで全部を送ってから次へ進むと、所要時間は往路と復路の和に
 * なり、実効帯域はおよそ半分になる。2 枚に分けると、チャンク N を宛先へ
 * 書いているあいだにチャンク N+1 を送信元から読める。大きいサイズの帯域は
 * 遅い方の片道（多くの場合は PCIe の片方向）に近づく。
 *
 * チャンクが転送サイズ以上のときは重ねる相手が無いので、結果は往復の直列
 * 転送になる。小さいサイズの数字が頭打ちなのはそのためで、不具合ではない。
 *
 * ---------------------------------------------------------------------------
 * Host ↔ VRAM の読み方
 * ---------------------------------------------------------------------------
 * ホスト側は hipHostMalloc でピン留めする。ページング可能な malloc だと
 * 実行時に余計なコピーが入り、リンクそのものの速度にならない。
 *
 * 表の列は次の意味を持つ。
 *   GPU 単独   相手 GPU は転送していない。その GPU の PCIe 片道の目安。
 *   同時 GPU   2 GPU が並行しているときの、その GPU 自身の転送時間から
 *              出した帯域。相手に帯域を取られて落ちると、ここに出る。
 *   同時合計   両 GPU が終わるまでの壁時計時間で、両 GPU の転送量の和を
 *              割ったシステム全体のスループット。
 *              同時 A と同時 B の和とは一致しない。遅い方に揃うまでの
 *              待ちが分母に入るため。
 *
 * 書き込みと読み出しは同時には流さない。読み出しの前に書き込みが一巡して
 * いるので、読むデータは VRAM 上にある。
 *
 * ---------------------------------------------------------------------------
 * 時間の取り方
 * ---------------------------------------------------------------------------
 * ホスト側の時計は使わない。各 GPU のストリームに hipEvent を置き、
 * hipEventElapsedTime で GPU 側の経過を取る。計測の前にウォームアップを
 * 数回走らせ、初回のページ割り当てやコマンドキューの遅れを本計測から外す。
 *
 * 2 本のストリームをまたぐ完了待ちには hipStreamWaitEvent を使う。
 * イベントを記録したデバイスと、経過を問い合わせるときのカレントデバイスは
 * 揃える。HIP はデバイスをまたいだ hipEventElapsedTime を保証しない。
 *
 * ---------------------------------------------------------------------------
 * バッファ
 * ---------------------------------------------------------------------------
 * 各 GPU に送信用と受信用の 2 面を確保する。双方向では
 *   A の送信バッファ → B の受信バッファ
 *   B の送信バッファ → A の受信バッファ
 * と別領域を使う。同じ領域を両方向が触ると、測定中にデータを壊し合う。
 * 送信バッファは計測中は書き換えない。最後の最大サイズ転送のあと、
 * 受信側の中身が送信側のパターンと一致するかを照合する。
 *
 * Host ↔ VRAM は GPU 間の検証が終わったあと、送信用バッファを再利用する。
 * ホスト側のパターンは GPU 間のものとシードを変えてあり、取り違えを
 * 不一致として検出できる。
 *
 * ---------------------------------------------------------------------------
 * 終了コード
 * ---------------------------------------------------------------------------
 * パターン照合がすべて一致すれば 0、1 ワードでも違えば 1。
 * HIP API が失敗したときはメッセージを出して直ちに終了する。
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* API の失敗は計測値として扱わない。呼び出し地点を残して終了する。 */
#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err_ = (call);                                              \
        if (err_ != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d — %s\n", __FILE__, __LINE__,     \
                    hipGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* 本計測の前に捨てる回数。最初の転送は割り当てとエンジン起動で遅い。 */
static constexpr int kWarmup = 3;
/*
 * ホスト経由 GPU 間転送の 1 チャンク。既定 8 MiB。
 * 小さすぎると投入のオーバーヘッドが帯域を削り、大きすぎると
 * 中程度のサイズでダブルバッファが効かなくなる。
 * 環境変数 P2P_CHUNK_MB で上書きできる。
 */
static constexpr size_t kDefaultChunk = 8ull << 20;

/* 1 サイズ分の結果。ms_each は 1 回転送、gbps は directions を含んだ帯域。 */
struct XferSample {
    size_t bytes;
    int iters;
    double ms_each;
    double gbps;
};

/*
 * 交渉済み PCIe の 1 ホップ。片方向ペイロードの単位は GB/s = 10^9 バイト/秒。
 * GT/s はレーンあたりの生レートで、符号化の分だけペイロードはそれより少ない。
 */
struct PcieHop {
    bool ok;
    char bdf[32];
    double gt;
    int lanes;
    int gen;
    double uni_gbps;
};

/* endpoint はデバイス直結。limit はルートまでの経路で最も遅いホップ。 */
struct PciePeak {
    bool ok;
    PcieHop endpoint;
    PcieHop limit;
    double gt;
    int lanes;
    int gen;
    double uni_gbps;
};

/*
 * 検証用の 32bit パターン。インデックスとシードから一意に再現できるので、
 * 期待値を別バッファに保存しなくてよい。ホストとデバイスで同じ式を使う。
 * 0x9E3779B9 は黄金比に由来する奇数で、下位ビットもそれなりに散らばる。
 */
__host__ __device__ static uint32_t word_pattern(size_t i, uint32_t seed) {
    return (uint32_t)i * 0x9E3779B9u ^ seed ^ (uint32_t)(i >> 32);
}

__global__ void fill_u32(uint32_t* p, size_t n, uint32_t seed) {
    /* グリッドより長いバッファはストライドで何周もする。 */
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        p[i] = word_pattern(i, seed);
    }
}

__global__ void check_u32(const uint32_t* p, size_t n, uint32_t seed, unsigned long long* bad) {
    /* 不一致は原子加算で数える。最初の 1 件で打ち切ると、壊れ方の規模が消える。 */
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (p[i] != word_pattern(i, seed)) atomicAdd(bad, 1ULL);
    }
}

static const char* link_name(uint32_t t) {
    switch (t) {
    case 0: return "HyperTransport";
    case 1: return "QPI";
    case 2: return "PCIe";
    case 3: return "InfiniBand";
    case 4: return "xGMI";
    default: return "不明";
    }
}

static void fmt_size(char* buf, size_t nbuf, size_t n) {
    if (n >= (1ull << 30) && (n % (1ull << 30)) == 0)
        snprintf(buf, nbuf, "%4zu GiB", n >> 30);
    else if (n >= (1ull << 20) && (n % (1ull << 20)) == 0)
        snprintf(buf, nbuf, "%4zu MiB", n >> 20);
    else if (n >= (1ull << 10) && (n % (1ull << 10)) == 0)
        snprintf(buf, nbuf, "%4zu KiB", n >> 10);
    else
        snprintf(buf, nbuf, "%4zu B", n);
}

static void fmt_time(char* buf, size_t nbuf, double ms) {
    if (ms < 1.0)
        snprintf(buf, nbuf, "%8.2f us", ms * 1000.0);
    else if (ms < 1000.0)
        snprintf(buf, nbuf, "%8.3f ms", ms);
    else
        snprintf(buf, nbuf, "%8.3f s ", ms / 1000.0);
}

/* ms ミリ秒で bytes バイト運んだときの GB/s。GB は 10^9。 */
static double gb_per_s(double bytes, double ms) {
    if (ms <= 0.0) return 0.0;
    return bytes / (ms * 1e-3) / 1e9;
}

/*
 * 交渉済み PCIe リンクの片方向ペイロード。単位は GB/s = 10^9 バイト/秒。
 * GT/s はレーンあたりの生レート。符号化の分だけペイロードはそれより少ない。
 */
static int pcie_gen_from_gt(double gt) {
    if (gt < 3.5) return 1;
    if (gt < 6.5) return 2;
    if (gt < 12.0) return 3;
    if (gt < 24.0) return 4;
    if (gt < 48.0) return 5;
    return 6;
}

static double pcie_uni_gbps(double gt, int lanes) {
    double eff;
    if (gt < 6.5) eff = 8.0 / 10.0;
    else if (gt < 48.0) eff = 128.0 / 130.0;
    else eff = 242.0 / 256.0;
    return gt * eff * (double)lanes / 8.0;
}

/* HIP の PCI アドレスから上流へ辿り、経路上で実測された最も遅いリンクを返す。 */
static bool pci_bdf_name(const char* s) {
    unsigned domain = 0, bus = 0, dev = 0, fn = 0;
    char tail = 0;
    return sscanf(s, "%x:%x:%x.%x%c", &domain, &bus, &dev, &fn, &tail) == 4;
}

static bool read_pcie_hop(const char* bdf, PcieHop& hop) {
    hop = {};
    char speed_path[160];
    char width_path[160];
    snprintf(speed_path, sizeof(speed_path), "/sys/bus/pci/devices/%s/current_link_speed", bdf);
    snprintf(width_path, sizeof(width_path), "/sys/bus/pci/devices/%s/current_link_width", bdf);

    FILE* fs = fopen(speed_path, "r");
    FILE* fw = fopen(width_path, "r");
    if (!fs || !fw) {
        if (fs) fclose(fs);
        if (fw) fclose(fw);
        return false;
    }

    char line[64] = {};
    const bool got_speed = fgets(line, (int)sizeof(line), fs) != nullptr;
    fclose(fs);
    int lanes = 0;
    const bool got_width = fscanf(fw, "%d", &lanes) == 1;
    fclose(fw);
    if (!got_speed || !got_width || lanes <= 0) return false;

    double gt = 0.0;
    if (sscanf(line, "%lf", &gt) != 1 || gt <= 0.0) return false;

    hop.ok = true;
    snprintf(hop.bdf, sizeof(hop.bdf), "%s", bdf);
    hop.gt = gt;
    hop.lanes = lanes;
    hop.gen = pcie_gen_from_gt(gt);
    hop.uni_gbps = pcie_uni_gbps(gt, lanes);
    return true;
}

static bool query_pcie_peak(int dev, PciePeak& out) {
    out = {};
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    char bdf[32] = {};
    if (hipDeviceGetPCIBusId(bdf, (int)sizeof(bdf), dev) != hipSuccess) {
        (void)hipGetLastError();
        snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.0",
                 prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    }

    char cur[32];
    snprintf(cur, sizeof(cur), "%s", bdf);
    bool any = false;
    /* 同じ速度のホップが続いたときは、より上流（ルートに近い側）を残す。 */
    for (int guard = 0; guard < 16 && pci_bdf_name(cur); guard++) {
        PcieHop hop{};
        if (read_pcie_hop(cur, hop)) {
            if (!any) out.endpoint = hop;
            if (!out.limit.ok || hop.uni_gbps <= out.limit.uni_gbps) out.limit = hop;
            any = true;
        }

        char syspath[160];
        snprintf(syspath, sizeof(syspath), "/sys/bus/pci/devices/%s", cur);
        char real[4096];
        if (!realpath(syspath, real)) break;
        char* slash = strrchr(real, '/');
        if (!slash || slash == real) break;
        *slash = '\0';
        const char* parent = strrchr(real, '/');
        parent = parent ? parent + 1 : real;
        if (!pci_bdf_name(parent) || strcmp(parent, cur) == 0) break;
        snprintf(cur, sizeof(cur), "%s", parent);
    }
    if (!any) return false;

    out.ok = true;
    out.gt = out.limit.gt;
    out.lanes = out.limit.lanes;
    out.gen = out.limit.gen;
    out.uni_gbps = out.limit.uni_gbps;
    return true;
}

static void print_theory(double measured, double theory) {
    if (theory > 0.0) {
        printf("   理論 %6.2f GB/s   効率 %5.1f%%", theory, 100.0 * measured / theory);
    } else {
        printf("   理論ピーク不明");
    }
    printf("\n");
}

/*
 * 反復回数。固定値が来ればそれを使う。
 * 自動のときは、そのサイズでおよそ 2 GiB 運ぶ回数にする。
 * 小さいサイズは 1 回が数マイクロ秒しかなく、イベントの分解能に負けるので
 * 下限を上げる。上限 500 は、ホストがコマンドを積む時間で計測が
 * 伸びすぎないようにする歯止め。
 */
static int choose_iters(size_t bytes, int fixed) {
    if (fixed > 0) return fixed;
    const size_t target = 2ull << 30;
    int n = (int)((target + bytes - 1) / bytes);
    if (bytes <= (64ull << 10)) n = std::max(n, 400);
    else if (bytes <= (1ull << 20)) n = std::max(n, 100);
    else n = std::max(n, 8);
    if (n > 500) n = 500;
    return n;
}

static void launch_fill(int dev, void* ptr, size_t bytes, uint32_t seed) {
    HIP_CHECK(hipSetDevice(dev));
    const size_t n = bytes / sizeof(uint32_t);
    fill_u32<<<1024, 256>>>(static_cast<uint32_t*>(ptr), n, seed);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

static unsigned long long launch_check(int dev, void* ptr, size_t bytes, uint32_t seed) {
    HIP_CHECK(hipSetDevice(dev));
    unsigned long long* d_bad = nullptr;
    HIP_CHECK(hipMalloc(&d_bad, sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(d_bad, 0, sizeof(unsigned long long)));
    const size_t n = bytes / sizeof(uint32_t);
    check_u32<<<1024, 256>>>(static_cast<const uint32_t*>(ptr), n, seed, d_bad);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    unsigned long long bad = 0;
    HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(bad), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_bad));
    return bad;
}

/*
 * dev から peer のメモリを直接見えるようにする。
 * 不可なら false。すでに有効なら成功扱いにする。
 * 有効化に失敗してもプロセスは止めない。呼び出し側がホスト経由へ切り替える。
 */
static bool enable_peer(int dev, int peer) {
    int can = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can, dev, peer));
    if (!can) return false;
    HIP_CHECK(hipSetDevice(dev));
    hipError_t err = hipDeviceEnablePeerAccess(peer, 0);
    if (err == hipErrorPeerAccessAlreadyEnabled) {
        (void)hipGetLastError();
        return true;
    }
    if (err != hipSuccess) {
        fprintf(stderr, "  hipDeviceEnablePeerAccess(%d ← %d) 失敗: %s\n",
                dev, peer, hipGetErrorString(err));
        (void)hipGetLastError();
        return false;
    }
    return true;
}

/*
 * ピアが使えないときの GPU → ホスト → GPU 経路。
 *
 * host[2] はピン留めしたチャンクバッファ。スロット 0 と 1 を交互に使う。
 * ready[i]    スロット i へのデバイス→ホストコピーが終わり、
 *             宛先 GPU が読んでよい。
 * consumed[i] 宛先がそのスロットを読み終わり、送信側が上書きしてよい。
 *
 * ストリームはデバイスに結び付く。s_src は送信 GPU、s_dst は宛先 GPU。
 * イベントも、記録するストリームと同じデバイスで作る。
 */
struct Pipe {
    int src;
    int dst;
    void* src_ptr;
    void* dst_ptr;
    hipStream_t s_src;
    hipStream_t s_dst;
    hipEvent_t ready[2];
    hipEvent_t consumed[2];
    void* host[2];
    size_t chunk;
};

static void pipe_init(Pipe& p, int src, int dst, void* src_ptr, void* dst_ptr, size_t chunk) {
    p.src = src;
    p.dst = dst;
    p.src_ptr = src_ptr;
    p.dst_ptr = dst_ptr;
    p.chunk = chunk;
    p.s_src = nullptr;
    p.s_dst = nullptr;
    p.host[0] = p.host[1] = nullptr;

    HIP_CHECK(hipSetDevice(src));
    HIP_CHECK(hipStreamCreate(&p.s_src));
    HIP_CHECK(hipSetDevice(dst));
    HIP_CHECK(hipStreamCreate(&p.s_dst));

    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipHostMalloc(&p.host[i], chunk));
        HIP_CHECK(hipSetDevice(src));
        HIP_CHECK(hipEventCreate(&p.ready[i]));
        HIP_CHECK(hipSetDevice(dst));
        HIP_CHECK(hipEventCreate(&p.consumed[i]));
        HIP_CHECK(hipEventRecord(p.consumed[i], p.s_dst));
    }
    HIP_CHECK(hipSetDevice(dst));
    HIP_CHECK(hipStreamSynchronize(p.s_dst));
}

static void pipe_destroy(Pipe& p) {
    for (int i = 0; i < 2; i++) {
        if (p.host[i]) HIP_CHECK(hipHostFree(p.host[i]));
        if (p.ready[i]) HIP_CHECK(hipEventDestroy(p.ready[i]));
        if (p.consumed[i]) HIP_CHECK(hipEventDestroy(p.consumed[i]));
        p.host[i] = nullptr;
        p.ready[i] = nullptr;
        p.consumed[i] = nullptr;
    }
    if (p.s_src) {
        HIP_CHECK(hipSetDevice(p.src));
        HIP_CHECK(hipStreamDestroy(p.s_src));
        p.s_src = nullptr;
    }
    if (p.s_dst) {
        HIP_CHECK(hipSetDevice(p.dst));
        HIP_CHECK(hipStreamDestroy(p.s_dst));
        p.s_dst = nullptr;
    }
}

/*
 * bytes バイトを 1 回、チャンクに切ってキューへ積む。ここでは待たない。
 *
 * 手順は次のとおり。
 *   1. スロット 0 へ、送信 VRAM から最初のチャンクを読む。
 *   2. 残りがあるあいだ、
 *        空いている側のスロットへ次のチャンクを読み始める（送信 GPU）。
 *        直前に読み終えたスロットを宛先 VRAM へ書く（宛先 GPU）。
 *   3. 最後のチャンクは読む相手が無いので、書くだけ。
 *
 * 同じスロットを再利用する前に consumed を待つ。待たないと、宛先が
 * まだ読んでいるホストメモリを送信側が上書きする。
 * ストリームは FIFO なので、反復を続けて積んでも前の反復の完了と
 * 次の反復の開始がこのイベントでつながる。ホストが反復のあいだに
 * 同期する必要はない。
 */
static void enqueue_staged(Pipe& p, size_t bytes) {
    size_t read_off = 0;
    size_t write_off = 0;
    size_t left_read = bytes;
    size_t left_write = bytes;

    auto d2h = [&](int slot, size_t off, size_t n) {
        HIP_CHECK(hipSetDevice(p.src));
        HIP_CHECK(hipStreamWaitEvent(p.s_src, p.consumed[slot], 0));
        HIP_CHECK(hipMemcpyAsync(static_cast<char*>(p.host[slot]),
                                 static_cast<char*>(p.src_ptr) + off, n,
                                 hipMemcpyDeviceToHost, p.s_src));
        HIP_CHECK(hipEventRecord(p.ready[slot], p.s_src));
    };
    auto h2d = [&](int slot, size_t off, size_t n) {
        HIP_CHECK(hipSetDevice(p.dst));
        HIP_CHECK(hipStreamWaitEvent(p.s_dst, p.ready[slot], 0));
        HIP_CHECK(hipMemcpyAsync(static_cast<char*>(p.dst_ptr) + off,
                                 static_cast<char*>(p.host[slot]), n,
                                 hipMemcpyHostToDevice, p.s_dst));
        HIP_CHECK(hipEventRecord(p.consumed[slot], p.s_dst));
    };

    const size_t n0 = std::min(p.chunk, left_read);
    d2h(0, read_off, n0);
    read_off += n0;
    left_read -= n0;

    int cur = 0;
    while (left_write > 0) {
        const int nxt = cur ^ 1;
        if (left_read > 0) {
            const size_t n2 = std::min(p.chunk, left_read);
            d2h(nxt, read_off, n2);
            read_off += n2;
            left_read -= n2;
        }
        const size_t n3 = std::min(p.chunk, left_write);
        h2d(cur, write_off, n3);
        write_off += n3;
        left_write -= n3;
        cur = nxt;
    }
}

static float elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

/*
 * directions は、計測窓のあいだに同時に流れていた転送の本数。
 * 単方向と単独は 1。GPU 間の双方向と Host の同時合計は 2。
 * 帯域 = (bytes × iters × directions) / 窓の時間。
 */
static XferSample finish_sample(size_t bytes, int iters, double total_ms, int directions) {
    XferSample s;
    s.bytes = bytes;
    s.iters = iters;
    s.ms_each = total_ms / iters;
    const double moved = (double)bytes * iters * directions;
    s.gbps = gb_per_s(moved, total_ms);
    return s;
}

/* ピアコピー 1 方向。コピーエンジンは送信側デバイスのストリームに載せる。 */
static XferSample bench_peer_uni(int src, int dst, void* src_ptr, void* dst_ptr,
                                 size_t bytes, int iters) {
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    HIP_CHECK(hipSetDevice(src));
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < kWarmup; i++) {
        HIP_CHECK(hipMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, bytes, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipSetDevice(src));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, bytes, stream));
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    const float ms = elapsed_ms(start, stop);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream));
    return finish_sample(bytes, iters, ms, 1);
}

/*
 * ピアコピー双方向。A→B と B→A を別ストリームに積み、両方が終わるまでを
 * 1 つの経過時間にする。バッファは方向ごとに別（a_tx→b_rx と b_tx→a_rx）。
 * 合計帯域なので finish_sample の directions は 2。
 */
static XferSample bench_peer_bi(int a, int b,
                                void* a_tx, void* b_rx,
                                void* b_tx, void* a_rx,
                                size_t bytes, int iters) {
    hipStream_t s0 = nullptr, s1 = nullptr;
    hipEvent_t start = nullptr, stop = nullptr, done1 = nullptr;
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipStreamCreate(&s0));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipStreamCreate(&s1));
    HIP_CHECK(hipEventCreate(&done1));

    for (int i = 0; i < kWarmup; i++) {
        HIP_CHECK(hipSetDevice(a));
        HIP_CHECK(hipMemcpyPeerAsync(b_rx, b, a_tx, a, bytes, s0));
        HIP_CHECK(hipSetDevice(b));
        HIP_CHECK(hipMemcpyPeerAsync(a_rx, a, b_tx, b, bytes, s1));
    }
    HIP_CHECK(hipStreamSynchronize(s0));
    HIP_CHECK(hipStreamSynchronize(s1));

    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipEventRecord(start, s0));
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipSetDevice(a));
        HIP_CHECK(hipMemcpyPeerAsync(b_rx, b, a_tx, a, bytes, s0));
        HIP_CHECK(hipSetDevice(b));
        HIP_CHECK(hipMemcpyPeerAsync(a_rx, a, b_tx, b, bytes, s1));
    }
    HIP_CHECK(hipEventRecord(done1, s1));
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipStreamWaitEvent(s0, done1, 0));
    HIP_CHECK(hipEventRecord(stop, s0));
    HIP_CHECK(hipEventSynchronize(stop));
    const float ms = elapsed_ms(start, stop);

    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(s0));
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipEventDestroy(done1));
    HIP_CHECK(hipStreamDestroy(s1));
    return finish_sample(bytes, iters, ms, 2);
}

/*
 * ホスト経由の単方向。開始イベントは送信ストリーム、終了は
 * 「宛先ストリームの最後の書き込み」を送信ストリームが待ったあと。
 * これで窓は、最初の読み出しから最後の書き込み完了までを含む。
 * 宛先側のイベントを送信側の hipEventElapsedTime に直接渡さない。
 */
static XferSample bench_staged_uni(Pipe& p, size_t bytes, int iters) {
    hipEvent_t start = nullptr, stop = nullptr, done = nullptr;
    HIP_CHECK(hipSetDevice(p.src));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipSetDevice(p.dst));
    HIP_CHECK(hipEventCreate(&done));

    for (int i = 0; i < kWarmup; i++) enqueue_staged(p, bytes);
    HIP_CHECK(hipSetDevice(p.src));
    HIP_CHECK(hipStreamSynchronize(p.s_src));
    HIP_CHECK(hipSetDevice(p.dst));
    HIP_CHECK(hipStreamSynchronize(p.s_dst));

    HIP_CHECK(hipSetDevice(p.src));
    HIP_CHECK(hipEventRecord(start, p.s_src));
    for (int i = 0; i < iters; i++) enqueue_staged(p, bytes);
    HIP_CHECK(hipSetDevice(p.dst));
    HIP_CHECK(hipEventRecord(done, p.s_dst));
    HIP_CHECK(hipSetDevice(p.src));
    HIP_CHECK(hipStreamWaitEvent(p.s_src, done, 0));
    HIP_CHECK(hipEventRecord(stop, p.s_src));
    HIP_CHECK(hipEventSynchronize(stop));
    const float ms = elapsed_ms(start, stop);

    HIP_CHECK(hipSetDevice(p.src));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipSetDevice(p.dst));
    HIP_CHECK(hipEventDestroy(done));
    return finish_sample(bytes, iters, ms, 1);
}

/*
 * ホスト経由の双方向。A→B 用パイプと B→A 用パイプを同時に積む。
 * ホストバッファはパイプごとに 2 枚あるので、方向どうしでメモリを共有しない。
 * 帯域は両方向の和。片方向の速度ではない。
 */
static XferSample bench_staged_bi(Pipe& ab, Pipe& ba, size_t bytes, int iters) {
    hipEvent_t start = nullptr, stop = nullptr, done_ab = nullptr, done_ba = nullptr;
    HIP_CHECK(hipSetDevice(ab.src));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipSetDevice(ab.dst));
    HIP_CHECK(hipEventCreate(&done_ab));
    HIP_CHECK(hipSetDevice(ba.dst));
    HIP_CHECK(hipEventCreate(&done_ba));

    for (int i = 0; i < kWarmup; i++) {
        enqueue_staged(ab, bytes);
        enqueue_staged(ba, bytes);
    }
    HIP_CHECK(hipStreamSynchronize(ab.s_src));
    HIP_CHECK(hipStreamSynchronize(ab.s_dst));
    HIP_CHECK(hipStreamSynchronize(ba.s_src));
    HIP_CHECK(hipStreamSynchronize(ba.s_dst));

    HIP_CHECK(hipSetDevice(ab.src));
    HIP_CHECK(hipEventRecord(start, ab.s_src));
    for (int i = 0; i < iters; i++) {
        enqueue_staged(ab, bytes);
        enqueue_staged(ba, bytes);
    }
    HIP_CHECK(hipSetDevice(ab.dst));
    HIP_CHECK(hipEventRecord(done_ab, ab.s_dst));
    HIP_CHECK(hipSetDevice(ba.dst));
    HIP_CHECK(hipEventRecord(done_ba, ba.s_dst));
    HIP_CHECK(hipSetDevice(ab.src));
    HIP_CHECK(hipStreamWaitEvent(ab.s_src, done_ab, 0));
    HIP_CHECK(hipStreamWaitEvent(ab.s_src, done_ba, 0));
    HIP_CHECK(hipEventRecord(stop, ab.s_src));
    HIP_CHECK(hipEventSynchronize(stop));
    const float ms = elapsed_ms(start, stop);

    HIP_CHECK(hipSetDevice(ab.src));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipSetDevice(ab.dst));
    HIP_CHECK(hipEventDestroy(done_ab));
    HIP_CHECK(hipSetDevice(ba.dst));
    HIP_CHECK(hipEventDestroy(done_ba));
    return finish_sample(bytes, iters, ms, 2);
}

/*
 * 1 GPU と、その GPU 専用のピン留めホストバッファ。
 * 2 GPU 同時計測でもホスト領域は分ける。同じページを両 GPU が叩くと、
 * リンクの速度ではなくホストメモリ上の競合を測ってしまう。
 */
struct DevHost {
    int dev;
    void* dev_ptr;   /* 既存の VRAM バッファを借りる。ここでは確保しない。 */
    void* host_ptr;  /* hipHostMalloc。キャッシュ可能で、読みにも書きにも使う。 */
    hipStream_t stream;
    uint32_t seed;
};

/*
 * 2 GPU 同時計測の 3 つの数字。
 *   a, b  各 GPU のストリーム上の開始から終了まで。相手の完了は待たない。
 *   sum   両方の転送が終わるまでの壁時計。directions = 2 の合計帯域。
 */
struct SimSample {
    XferSample a;
    XferSample b;
    XferSample sum;
};

static void fill_host_pattern(void* host, size_t bytes, uint32_t seed) {
    auto* p = static_cast<uint32_t*>(host);
    const size_t n = bytes / sizeof(uint32_t);
    for (size_t i = 0; i < n; i++) p[i] = word_pattern(i, seed);
}

static void devhost_init(DevHost& h, int dev, void* dev_ptr, size_t bytes, uint32_t seed) {
    h.dev = dev;
    h.dev_ptr = dev_ptr;
    h.host_ptr = nullptr;
    h.stream = nullptr;
    h.seed = seed;
    HIP_CHECK(hipHostMalloc(&h.host_ptr, bytes));
    fill_host_pattern(h.host_ptr, bytes, seed);
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreate(&h.stream));
}

static void devhost_destroy(DevHost& h) {
    if (h.stream) {
        HIP_CHECK(hipSetDevice(h.dev));
        HIP_CHECK(hipStreamDestroy(h.stream));
        h.stream = nullptr;
    }
    if (h.host_ptr) {
        HIP_CHECK(hipHostFree(h.host_ptr));
        h.host_ptr = nullptr;
    }
}

static void enqueue_host(DevHost& h, size_t bytes, bool to_device) {
    HIP_CHECK(hipSetDevice(h.dev));
    if (to_device) {
        HIP_CHECK(hipMemcpyAsync(h.dev_ptr, h.host_ptr, bytes, hipMemcpyHostToDevice, h.stream));
    } else {
        HIP_CHECK(hipMemcpyAsync(h.host_ptr, h.dev_ptr, bytes, hipMemcpyDeviceToHost, h.stream));
    }
}

/*
 * 1 GPU だけの Host↔VRAM。
 * to_device が真なら書き込み (HtoD)、偽なら読み出し (DtoH)。
 * 相手 GPU のストリームには何も積まない。
 */
static XferSample bench_host(DevHost& h, size_t bytes, int iters, bool to_device) {
    hipEvent_t start = nullptr, stop = nullptr;
    HIP_CHECK(hipSetDevice(h.dev));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < kWarmup; i++) enqueue_host(h, bytes, to_device);
    HIP_CHECK(hipStreamSynchronize(h.stream));

    HIP_CHECK(hipSetDevice(h.dev));
    HIP_CHECK(hipEventRecord(start, h.stream));
    for (int i = 0; i < iters; i++) enqueue_host(h, bytes, to_device);
    HIP_CHECK(hipEventRecord(stop, h.stream));
    HIP_CHECK(hipEventSynchronize(stop));
    const float ms = elapsed_ms(start, stop);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return finish_sample(bytes, iters, ms, 1);
}

/*
 * 2 GPU が同じ方向を同時に転送する。
 *
 * 各 GPU の stop はその GPU のコピーが終わった時点。相手は待たない。
 * そのため「同時 GPU0」「同時 GPU1」は、競合した状態での各リンクの速度。
 *
 * wall_stop は GPU A のストリーム上で、GPU B の stop を待ってから記録する。
 * A の開始から wall_stop までが「両方が終わるまで」の窓で、同時合計の分母。
 * B の経過を A のイベントと比較はしない。B の経過は B のイベント対だけで取る。
 */
static SimSample bench_host_both(DevHost& a, DevHost& b, size_t bytes, int iters, bool to_device) {
    hipEvent_t start_a = nullptr, stop_a = nullptr, wall_stop = nullptr;
    hipEvent_t start_b = nullptr, stop_b = nullptr;
    HIP_CHECK(hipSetDevice(a.dev));
    HIP_CHECK(hipEventCreate(&start_a));
    HIP_CHECK(hipEventCreate(&stop_a));
    HIP_CHECK(hipEventCreate(&wall_stop));
    HIP_CHECK(hipSetDevice(b.dev));
    HIP_CHECK(hipEventCreate(&start_b));
    HIP_CHECK(hipEventCreate(&stop_b));

    for (int i = 0; i < kWarmup; i++) {
        enqueue_host(a, bytes, to_device);
        enqueue_host(b, bytes, to_device);
    }
    HIP_CHECK(hipStreamSynchronize(a.stream));
    HIP_CHECK(hipStreamSynchronize(b.stream));

    HIP_CHECK(hipSetDevice(a.dev));
    HIP_CHECK(hipEventRecord(start_a, a.stream));
    HIP_CHECK(hipSetDevice(b.dev));
    HIP_CHECK(hipEventRecord(start_b, b.stream));
    for (int i = 0; i < iters; i++) {
        enqueue_host(a, bytes, to_device);
        enqueue_host(b, bytes, to_device);
    }
    HIP_CHECK(hipSetDevice(a.dev));
    HIP_CHECK(hipEventRecord(stop_a, a.stream));
    HIP_CHECK(hipSetDevice(b.dev));
    HIP_CHECK(hipEventRecord(stop_b, b.stream));
    HIP_CHECK(hipSetDevice(a.dev));
    HIP_CHECK(hipStreamWaitEvent(a.stream, stop_b, 0));
    HIP_CHECK(hipEventRecord(wall_stop, a.stream));
    HIP_CHECK(hipEventSynchronize(wall_stop));

    HIP_CHECK(hipSetDevice(a.dev));
    const float ms_a = elapsed_ms(start_a, stop_a);
    const float ms_wall = elapsed_ms(start_a, wall_stop);
    HIP_CHECK(hipSetDevice(b.dev));
    const float ms_b = elapsed_ms(start_b, stop_b);

    HIP_CHECK(hipSetDevice(a.dev));
    HIP_CHECK(hipEventDestroy(start_a));
    HIP_CHECK(hipEventDestroy(stop_a));
    HIP_CHECK(hipEventDestroy(wall_stop));
    HIP_CHECK(hipSetDevice(b.dev));
    HIP_CHECK(hipEventDestroy(start_b));
    HIP_CHECK(hipEventDestroy(stop_b));

    SimSample s;
    s.a = finish_sample(bytes, iters, ms_a, 1);
    s.b = finish_sample(bytes, iters, ms_b, 1);
    s.sum = finish_sample(bytes, iters, ms_wall, 2);
    return s;
}

/* 読み出したホストメモリを CPU で全ワード照合する。既定の上限は 16 GiB。 */
static bool verify_host_pattern(const char* label, void* host, size_t bytes, uint32_t seed) {
    const auto* p = static_cast<const uint32_t*>(host);
    const size_t n = bytes / sizeof(uint32_t);
    unsigned long long bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (p[i] != word_pattern(i, seed)) bad++;
    }
    if (bad == 0) {
        printf("  %-18s %s\n", label, "OK");
        return true;
    }
    printf("  %-18s 不一致 %llu ワード\n", label, bad);
    return false;
}

struct Peak {
    double gbps;
    size_t bytes;
};

static void consider_peak(Peak& p, double gbps, size_t bytes) {
    if (gbps > p.gbps) {
        p.gbps = gbps;
        p.bytes = bytes;
    }
}

static void print_host_header(int gpu_a, int gpu_b) {
    char c0[32], c1[32], s0[32], s1[32];
    snprintf(c0, sizeof(c0), "GPU%d 単独", gpu_a);
    snprintf(c1, sizeof(c1), "GPU%d 単独", gpu_b);
    snprintf(s0, sizeof(s0), "同時 GPU%d", gpu_a);
    snprintf(s1, sizeof(s1), "同時 GPU%d", gpu_b);
    printf("  %-10s %6s  %14s  %14s  %14s  %14s  %14s\n",
           "サイズ", "反復", c0, c1, s0, s1, "同時合計");
    printf("  ---------- ------  --------------  --------------  --------------  --------------  --------------\n");
}

static void print_host_row(size_t bytes, int iters, double solo_a, double solo_b,
                           double sim_a, double sim_b, double sim_sum) {
    char sz[32];
    fmt_size(sz, sizeof(sz), bytes);
    printf("  %s %6d  %11.2f GB/s  %11.2f GB/s  %11.2f GB/s  %11.2f GB/s  %11.2f GB/s\n",
           sz, iters, solo_a, solo_b, sim_a, sim_b, sim_sum);
    fflush(stdout);
}

static void print_header_row() {
    printf("  %-10s %6s  %12s  %12s\n", "サイズ", "反復", "1回転送", "帯域");
    printf("  ---------- ------  ------------  ------------\n");
}

static void print_sample(const XferSample& s) {
    char sz[32], tm[32];
    fmt_size(sz, sizeof(sz), s.bytes);
    fmt_time(tm, sizeof(tm), s.ms_each);
    printf("  %s %6d  %12s  %8.2f GB/s\n", sz, s.iters, tm, s.gbps);
    fflush(stdout);
}

static const XferSample* peak_of(const std::vector<XferSample>& v) {
    const XferSample* best = &v[0];
    for (const auto& s : v) {
        if (s.gbps > best->gbps) best = &s;
    }
    return best;
}

static void print_devices() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    printf("=== デバイス (%d) ===\n", count);
    for (int d = 0; d < count; d++) {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, d));
        HIP_CHECK(hipSetDevice(d));
        size_t free_b = 0, total_b = 0;
        HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
        char pci[32] = {};
        hipError_t pci_err = hipDeviceGetPCIBusId(pci, (int)sizeof(pci), d);
        if (pci_err != hipSuccess) {
            snprintf(pci, sizeof(pci), "%04x:%02x:%02x",
                     prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
            (void)hipGetLastError();
        }
        printf("  [%d] %s\n", d, prop.name);
        printf("      arch %s | CU %d | %d MHz | VRAM %.1f GB (空き %.1f GB) | PCI %s\n",
               prop.gcnArchName, prop.multiProcessorCount, prop.clockRate / 1000,
               total_b / (1024.0 * 1024.0 * 1024.0),
               free_b / (1024.0 * 1024.0 * 1024.0), pci);
    }
    printf("\n");
}

static std::vector<size_t> make_sizes(size_t nbytes) {
    std::vector<size_t> sizes;
    for (size_t s = 4ull << 10; s < nbytes; s *= 4) sizes.push_back(s);
    if (sizes.empty() || sizes.back() != nbytes) sizes.push_back(nbytes);
    return sizes;
}

static bool verify_copy(const char* label, int dev, void* ptr, size_t bytes, uint32_t seed) {
    const unsigned long long bad = launch_check(dev, ptr, bytes, seed);
    if (bad == 0) {
        printf("  %-18s %s\n", label, "OK");
        return true;
    }
    printf("  %-18s 不一致 %llu ワード\n", label, bad);
    return false;
}

/*
 * 計測の順序:
 *   1. デバイスとリンク、ピアアクセスの可否を表示する。
 *   2. 空き VRAM に収まる最大サイズを決め、各 GPU に送受 2 面を確保する。
 *   3. GPU 間を単方向 2 本、続いて双方向で測る。
 *   4. 最大サイズの受信バッファをパターン照合する。
 *   5. 送信用 VRAM を Host↔VRAM に転用し、書き込み、次に読み出しを測る。
 *      各サイズで単独 A、単独 B、同時の順。最後の同時転送がバッファ全体を
 *      覆うので、そのあと全ワードを照合できる。
 *   6. サイズ横断のピークをサマリに出す。
 *
 * サイズ列は 4 KiB から 4 倍ずつ増やし、最後をちょうど最大サイズにする。
 * 4 KiB 未満はコピーエンジンの起動コストが支配的で、帯域の比較に向かない。
 */
int main(int argc, char** argv) {
    int gpu_a = 0;
    int gpu_b = 1;
    unsigned long long max_mb = 16384;
    int fixed_iters = 0;
    if (argc >= 3) {
        gpu_a = atoi(argv[1]);
        gpu_b = atoi(argv[2]);
    }
    if (argc >= 4) max_mb = strtoull(argv[3], nullptr, 10);
    if (argc >= 5) fixed_iters = atoi(argv[4]);
    if (max_mb == 0) max_mb = 1;

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║   ROCm 2-GPU VRAM 転送帯域                   ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    print_devices();

    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    if (count < 2) {
        fprintf(stderr, "GPU が %d 枚です。2 枚以上必要です。\n", count);
        return 1;
    }
    if (gpu_a == gpu_b || gpu_a < 0 || gpu_b < 0 || gpu_a >= count || gpu_b >= count) {
        fprintf(stderr, "GPU 番号が不正です: %d と %d (有効範囲 0..%d、互いに異なる番号)\n",
                gpu_a, gpu_b, count - 1);
        return 1;
    }

    printf("=== リンク GPU %d ↔ GPU %d ===\n", gpu_a, gpu_b);
    uint32_t link = 0, hops = 0;
    hipError_t link_err = hipExtGetLinkTypeAndHopCount(gpu_a, gpu_b, &link, &hops);
    if (link_err == hipSuccess && link != 0xFFFFFFFFu) {
        printf("  種類: %s   hop: %u\n", link_name(link), hops);
    } else if (link_err == hipSuccess) {
        printf("  種類: 直接リンクなし\n");
    } else {
        printf("  種類: 取得できません (%s)\n", hipGetErrorString(link_err));
        (void)hipGetLastError();
    }
    int rank_ab = -1, rank_ba = -1;
    if (hipDeviceGetP2PAttribute(&rank_ab, hipDevP2PAttrPerformanceRank, gpu_a, gpu_b) != hipSuccess) {
        (void)hipGetLastError();
        rank_ab = -1;
    }
    if (hipDeviceGetP2PAttribute(&rank_ba, hipDevP2PAttrPerformanceRank, gpu_b, gpu_a) != hipSuccess) {
        (void)hipGetLastError();
        rank_ba = -1;
    }

    const bool peer_ab = enable_peer(gpu_a, gpu_b);
    const bool peer_ba = enable_peer(gpu_b, gpu_a);
    printf("  ピアアクセス: %d → %d %s", gpu_a, gpu_b, peer_ab ? "可" : "不可");
    if (rank_ab >= 0) printf("  (perfRank %d)", rank_ab);
    printf("\n");
    printf("  ピアアクセス: %d → %d %s", gpu_b, gpu_a, peer_ba ? "可" : "不可");
    if (rank_ba >= 0) printf("  (perfRank %d)", rank_ba);
    printf("\n");

    bool use_peer = peer_ab && peer_ba;
    if (use_peer) {
        printf("  計測方式: hipMemcpyPeerAsync（VRAM 間の直接転送）\n\n");
    } else {
        printf("  計測方式: ピン留めホスト経由のダブルバッファ\n");
        printf("            （この組ではデバイス間の直接ピアアクセスを使えません）\n\n");
    }

    size_t free_a = 0, total_a = 0, free_b = 0, total_b = 0;
    HIP_CHECK(hipSetDevice(gpu_a));
    HIP_CHECK(hipMemGetInfo(&free_a, &total_a));
    HIP_CHECK(hipSetDevice(gpu_b));
    HIP_CHECK(hipMemGetInfo(&free_b, &total_b));

    /*
     * 各 GPU に送信と受信の 2 面が要る。空きから予約分を引き、その半分が
     * 1 面の上限。256 バイト未満の端数は切り捨て、パターン照合のワード境界
     * とコピーのアライメントを揃える。予約はランタイムと画面用の余裕。
     */
    const size_t reserve = 512ull << 20;
    size_t budget = std::min(free_a, free_b);
    if (budget > reserve) budget -= reserve;
    else budget /= 2;
    size_t nbytes = max_mb << 20;
    const size_t fit = budget / 2;
    if (nbytes > fit) {
        printf("  要求 %llu MiB × 2 バッファは空き VRAM に収まらないため、%zu MiB に縮小します。\n\n",
               max_mb, fit >> 20);
        nbytes = fit & ~size_t{255};
    }
    nbytes &= ~size_t{255};
    if (nbytes < (4ull << 10)) {
        fprintf(stderr, "計測に使える VRAM が足りません。\n");
        return 1;
    }

    /*
     * a_tx / b_tx は送信専用。計測ループでは読まれるだけで、中のパターンは
     * 最後まで残る。a_rx / b_rx は受信専用で、計測前はゼロ。
     * 双方向でもこの 4 面は役割が重ならない。
     */
    void* a_tx = nullptr;
    void* a_rx = nullptr;
    void* b_tx = nullptr;
    void* b_rx = nullptr;
    HIP_CHECK(hipSetDevice(gpu_a));
    HIP_CHECK(hipMalloc(&a_tx, nbytes));
    HIP_CHECK(hipMalloc(&a_rx, nbytes));
    HIP_CHECK(hipSetDevice(gpu_b));
    HIP_CHECK(hipMalloc(&b_tx, nbytes));
    HIP_CHECK(hipMalloc(&b_rx, nbytes));

    const uint32_t seed_a = 0xA11A0001u;
    const uint32_t seed_b = 0xB22B0002u;
    launch_fill(gpu_a, a_tx, nbytes, seed_a);
    launch_fill(gpu_b, b_tx, nbytes, seed_b);
    HIP_CHECK(hipSetDevice(gpu_a));
    HIP_CHECK(hipMemset(a_rx, 0, nbytes));
    HIP_CHECK(hipSetDevice(gpu_b));
    HIP_CHECK(hipMemset(b_rx, 0, nbytes));

    size_t chunk = kDefaultChunk;
    if (const char* env = getenv("P2P_CHUNK_MB")) {
        unsigned long v = strtoul(env, nullptr, 10);
        if (v >= 1 && v <= 256) chunk = (size_t)v << 20;
    }
    if (chunk > nbytes) chunk = nbytes;

    Pipe pipe_ab{};
    Pipe pipe_ba{};
    if (!use_peer) {
        pipe_init(pipe_ab, gpu_a, gpu_b, a_tx, b_rx, chunk);
        pipe_init(pipe_ba, gpu_b, gpu_a, b_tx, a_rx, chunk);
    } else {
        /*
         * canAccessPeer が真でも、実際のコピーが失敗するプラットフォームがある。
         * 1 MiB で試し、失敗したら同じホスト経由パイプに落とす。
         */
        hipError_t probe = hipMemcpyPeer(b_rx, gpu_b, a_tx, gpu_a, std::min(nbytes, (size_t)1 << 20));
        if (probe != hipSuccess) {
            fprintf(stderr, "ピアコピーの試行が失敗したためホスト経由に切り替えます: %s\n",
                    hipGetErrorString(probe));
            (void)hipGetLastError();
            use_peer = false;
            pipe_init(pipe_ab, gpu_a, gpu_b, a_tx, b_rx, chunk);
            pipe_init(pipe_ba, gpu_b, gpu_a, b_tx, a_rx, chunk);
        }
    }

    printf("=== 設定 ===\n");
    printf("  方向:       GPU %d ↔ GPU %d\n", gpu_a, gpu_b);
    printf("  最大サイズ: %.1f MiB（各 GPU に送信用・受信用の 2 バッファ）\n",
           nbytes / (1024.0 * 1024.0));
    printf("  反復:       %s\n", fixed_iters > 0 ? "固定" : "サイズごとに自動");
    if (!use_peer) printf("  チャンク:   %zu MiB × 2（ダブルバッファ）\n", chunk >> 20);
    printf("  帯域単位:   GB/s = 10^9 B/s。GPU 間の双方向と Host 同時実行は合計\n");
    printf("  Host↔VRAM:  ピン留めメモリ。各 GPU の単独と、2 GPU 同時の読み書き\n\n");

    const std::vector<size_t> sizes = make_sizes(nbytes);

    auto run_uni = [&](int src, int dst, void* src_ptr, void* dst_ptr, Pipe* pipe) {
        std::vector<XferSample> out;
        out.reserve(sizes.size());
        for (size_t bytes : sizes) {
            const int iters = choose_iters(bytes, fixed_iters);
            XferSample s = use_peer ? bench_peer_uni(src, dst, src_ptr, dst_ptr, bytes, iters)
                                    : bench_staged_uni(*pipe, bytes, iters);
            print_sample(s);
            out.push_back(s);
        }
        return out;
    };

    printf("=== 単方向  GPU %d → GPU %d ===\n", gpu_a, gpu_b);
    print_header_row();
    const auto a_to_b = run_uni(gpu_a, gpu_b, a_tx, b_rx, &pipe_ab);
    printf("\n");

    printf("=== 単方向  GPU %d → GPU %d ===\n", gpu_b, gpu_a);
    print_header_row();
    const auto b_to_a = run_uni(gpu_b, gpu_a, b_tx, a_rx, &pipe_ba);
    printf("\n");

    printf("=== 双方向  GPU %d ↔ GPU %d（同時） ===\n", gpu_a, gpu_b);
    print_header_row();
    std::vector<XferSample> both;
    both.reserve(sizes.size());
    for (size_t bytes : sizes) {
        const int iters = choose_iters(bytes, fixed_iters);
        XferSample s = use_peer ? bench_peer_bi(gpu_a, gpu_b, a_tx, b_rx, b_tx, a_rx, bytes, iters)
                                : bench_staged_bi(pipe_ab, pipe_ba, bytes, iters);
        print_sample(s);
        both.push_back(s);
    }
    printf("\n");

    printf("=== 検証（GPU 間・最大サイズ） ===\n");
    const bool ok_ab = verify_copy("GPU A → B", gpu_b, b_rx, nbytes, seed_a);
    const bool ok_ba = verify_copy("GPU B → A", gpu_a, a_rx, nbytes, seed_b);
    printf("\n");

    /*
     * GPU 間の照合が済んだので、送信バッファを Host↔VRAM に貸し出す。
     * シードは GPU 間のものと別。読み出しは直前の書き込みが VRAM に置いた
     * パターンをホストへ戻すので、書き込みを先にすべて終わらせる。
     */
    const uint32_t seed_ha = 0xC33C1001u;
    const uint32_t seed_hb = 0xD44D2002u;
    DevHost host_a{};
    DevHost host_b{};
    devhost_init(host_a, gpu_a, a_tx, nbytes, seed_ha);
    devhost_init(host_b, gpu_b, b_tx, nbytes, seed_hb);

    Peak peak_wa{0, 0}, peak_wb{0, 0}, peak_wsum{0, 0};
    Peak peak_ra{0, 0}, peak_rb{0, 0}, peak_rsum{0, 0};

    auto run_host_dir = [&](bool to_device, Peak& pa, Peak& pb, Peak& psum) {
        print_host_header(gpu_a, gpu_b);
        for (size_t bytes : sizes) {
            const int iters = choose_iters(bytes, fixed_iters);
            const XferSample solo_a = bench_host(host_a, bytes, iters, to_device);
            const XferSample solo_b = bench_host(host_b, bytes, iters, to_device);
            const SimSample sim = bench_host_both(host_a, host_b, bytes, iters, to_device);
            print_host_row(bytes, iters, solo_a.gbps, solo_b.gbps,
                           sim.a.gbps, sim.b.gbps, sim.sum.gbps);
            consider_peak(pa, solo_a.gbps, bytes);
            consider_peak(pb, solo_b.gbps, bytes);
            consider_peak(psum, sim.sum.gbps, bytes);
        }
        printf("\n");
    };

    printf("=== Host → VRAM 書き込み ===\n");
    printf("  単独は相手 GPU が遊んでいるときの帯域。同時は 2 GPU が並行して書き込む。\n");
    printf("  同時 GPU は各 GPU の転送時間、同時合計は両方が終わるまでの合計帯域。\n");
    run_host_dir(true, peak_wa, peak_wb, peak_wsum);

    printf("=== VRAM → Host 読み出し ===\n");
    printf("  単独は相手 GPU が遊んでいるときの帯域。同時は 2 GPU が並行して読み出す。\n");
    run_host_dir(false, peak_ra, peak_rb, peak_rsum);

    printf("=== 検証（Host ↔ VRAM・最大サイズ） ===\n");
    const bool ok_wh_a = verify_copy("書込 GPU A", gpu_a, a_tx, nbytes, seed_ha);
    const bool ok_wh_b = verify_copy("書込 GPU B", gpu_b, b_tx, nbytes, seed_hb);
    const bool ok_rh_a = verify_host_pattern("読出 GPU A", host_a.host_ptr, nbytes, seed_ha);
    const bool ok_rh_b = verify_host_pattern("読出 GPU B", host_b.host_ptr, nbytes, seed_hb);
    printf("\n");
    const bool ok_host = ok_wh_a && ok_wh_b && ok_rh_a && ok_rh_b;

    const XferSample* peak_ab = peak_of(a_to_b);
    const XferSample* peak_ba = peak_of(b_to_a);
    const XferSample* peak_bi = peak_of(both);
    char sz_ab[32], sz_ba[32], sz_bi[32];
    char sz_wa[32], sz_wb[32], sz_ws[32], sz_ra[32], sz_rb[32], sz_rs[32];
    fmt_size(sz_ab, sizeof(sz_ab), peak_ab->bytes);
    fmt_size(sz_ba, sizeof(sz_ba), peak_ba->bytes);
    fmt_size(sz_bi, sizeof(sz_bi), peak_bi->bytes);
    fmt_size(sz_wa, sizeof(sz_wa), peak_wa.bytes);
    fmt_size(sz_wb, sizeof(sz_wb), peak_wb.bytes);
    fmt_size(sz_ws, sizeof(sz_ws), peak_wsum.bytes);
    fmt_size(sz_ra, sizeof(sz_ra), peak_ra.bytes);
    fmt_size(sz_rb, sizeof(sz_rb), peak_rb.bytes);
    fmt_size(sz_rs, sizeof(sz_rs), peak_rsum.bytes);

    printf("=== サマリ ===\n");
    printf("  GPU 間の方式:         %s\n",
           use_peer ? "ピア直接 (hipMemcpyPeerAsync)" : "ホスト経由ダブルバッファ");

    PciePeak pcie_a{};
    PciePeak pcie_b{};
    const bool pcie_a_ok = query_pcie_peak(gpu_a, pcie_a);
    const bool pcie_b_ok = query_pcie_peak(gpu_b, pcie_b);
    const bool xgmi = (link_err == hipSuccess && link == 4);
    double theory_ab = 0.0, theory_ba = 0.0, theory_bi = 0.0;
    const double theory_wa = pcie_a_ok ? pcie_a.uni_gbps : 0.0;
    const double theory_wb = pcie_b_ok ? pcie_b.uni_gbps : 0.0;
    const double theory_host_sum = (pcie_a_ok && pcie_b_ok) ? pcie_a.uni_gbps + pcie_b.uni_gbps : 0.0;
    if (!xgmi && pcie_a_ok && pcie_b_ok) {
        const double slower = std::min(pcie_a.uni_gbps, pcie_b.uni_gbps);
        theory_ab = slower;
        theory_ba = slower;
        theory_bi = slower * 2.0;
    }

    printf("  理論ピークの基準:     GPU からルートポートまでの経路で、実測された最も遅いホップ\n");
    auto print_pcie = [](int gpu, const PciePeak& p, bool ok) {
        if (!ok) {
            printf("    GPU %d               リンク速度を取得できません\n", gpu);
            return;
        }
        printf("    GPU %d  経路の最遅 %s  PCIe %d.0 x%d  %.1f GT/s  片方向 %.2f GB/s\n",
               gpu, p.limit.bdf, p.gen, p.lanes, p.gt, p.uni_gbps);
        const bool same = p.endpoint.ok && p.endpoint.gen == p.limit.gen &&
                          p.endpoint.lanes == p.limit.lanes &&
                          p.endpoint.gt == p.limit.gt;
        if (!same && p.endpoint.ok) {
            printf("           デバイス直結 %s  PCIe %d.0 x%d  %.1f GT/s  片方向 %.2f GB/s\n",
                   p.endpoint.bdf, p.endpoint.gen, p.endpoint.lanes,
                   p.endpoint.gt, p.endpoint.uni_gbps);
        }
    };
    print_pcie(gpu_a, pcie_a, pcie_a_ok);
    print_pcie(gpu_b, pcie_b, pcie_b_ok);
    if (xgmi) {
        printf("    GPU 間は xGMI のため、PCIe では上限を置かない\n");
    } else {
        printf("    GPU 間の片方向は両経路の遅い方、双方向合計はその 2 倍\n");
        printf("    Host の単独はその GPU の最遅ホップ、同時合計は両経路の和\n");
    }

    printf("  ピーク %d → %d:      %6.2f GB/s @ %s", gpu_a, gpu_b, peak_ab->gbps, sz_ab);
    print_theory(peak_ab->gbps, theory_ab);
    printf("  ピーク %d → %d:      %6.2f GB/s @ %s", gpu_b, gpu_a, peak_ba->gbps, sz_ba);
    print_theory(peak_ba->gbps, theory_ba);
    printf("  ピーク双方向合計:    %6.2f GB/s @ %s", peak_bi->gbps, sz_bi);
    print_theory(peak_bi->gbps, theory_bi);
    printf("  書込 単独 GPU %d:    %6.2f GB/s @ %s", gpu_a, peak_wa.gbps, sz_wa);
    print_theory(peak_wa.gbps, theory_wa);
    printf("  書込 単独 GPU %d:    %6.2f GB/s @ %s", gpu_b, peak_wb.gbps, sz_wb);
    print_theory(peak_wb.gbps, theory_wb);
    printf("  書込 同時合計:       %6.2f GB/s @ %s", peak_wsum.gbps, sz_ws);
    print_theory(peak_wsum.gbps, theory_host_sum);
    printf("  読出 単独 GPU %d:    %6.2f GB/s @ %s", gpu_a, peak_ra.gbps, sz_ra);
    print_theory(peak_ra.gbps, theory_wa);
    printf("  読出 単独 GPU %d:    %6.2f GB/s @ %s", gpu_b, peak_rb.gbps, sz_rb);
    print_theory(peak_rb.gbps, theory_wb);
    printf("  読出 同時合計:       %6.2f GB/s @ %s", peak_rsum.gbps, sz_rs);
    print_theory(peak_rsum.gbps, theory_host_sum);
    printf("  検証:                 %s\n\n", (ok_ab && ok_ba && ok_host) ? "OK" : "失敗");

    devhost_destroy(host_b);
    devhost_destroy(host_a);

    if (!use_peer) {
        pipe_destroy(pipe_ba);
        pipe_destroy(pipe_ab);
    }
    HIP_CHECK(hipSetDevice(gpu_b));
    HIP_CHECK(hipFree(b_tx));
    HIP_CHECK(hipFree(b_rx));
    HIP_CHECK(hipSetDevice(gpu_a));
    HIP_CHECK(hipFree(a_tx));
    HIP_CHECK(hipFree(a_rx));

    printf("=== 完了 ===\n");
    return (ok_ab && ok_ba && ok_host) ? 0 : 1;
}

/**
 * Bölge Seed Verisi — node_map ve graph_edges
 * ==============================================
 * Bu veriler artık DB'de (RegionConfig.nodeMap / RegionConfig.graphEdges) tutuluyor;
 * burada sadece server.ts başlangıç seed'i için bir kerelik referans olarak duruyor.
 * Kod içinde şehir bazlı hiçbir mantık (if/else) bu dosyaya bağlı değildir —
 * sadece ilk kurulumda DB'ye yazılacak veriler.
 */
import type { NodeMap, GraphEdges } from './python-model.service'

// Kayseri İldem: 34 node (9 kavşak × A/B/C/D kolları, bazılarında eksik kol var)
export const KAYSERI_ILDEM_NODE_MAP: NodeMap = {
  89:  { A: 0,  B: 1,  C: 2,  D: 3 },
  187: { A: 4,  B: 5,  C: 6,  D: 7 },
  95:  { A: 8,  B: 9,  C: 10, D: 11 },
  121: { A: 12, B: 13, C: 14, D: 15 },
  184: { A: 16, B: 17,        D: 18 },  // C yok
  188: { A: 19, B: 20, C: 21, D: 22 },
  117: { A: 23,        C: 24, D: 25 },  // B yok
  192: { A: 26, B: 27, C: 28, D: 29 },
  194: { A: 30, B: 31, C: 32, D: 33 },
}

// data/Kayseri/kol_bazli_graph.csv içeriği (from,to)
export const KAYSERI_ILDEM_GRAPH_EDGES: GraphEdges = [
  [0, 4], [0, 5], [0, 6], [1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6],
  [0, 7], [0, 8], [0, 9], [0, 10], [1, 7], [1, 8], [1, 9], [1, 10], [2, 7], [2, 8], [2, 9], [2, 10],
  [3, 7], [3, 8], [3, 9], [3, 10], [0, 11], [0, 12], [0, 13], [0, 14], [1, 11], [1, 12], [1, 13], [1, 14],
  [2, 11], [2, 12], [2, 13], [2, 14], [3, 11], [3, 12], [3, 13], [3, 14], [4, 7], [4, 8], [4, 9], [4, 10],
  [5, 7], [5, 8], [5, 9], [5, 10], [6, 7], [6, 8], [6, 9], [6, 10], [4, 11], [4, 12], [4, 13], [4, 14],
  [5, 11], [5, 12], [5, 13], [5, 14], [6, 11], [6, 12], [6, 13], [6, 14], [7, 11], [7, 12], [7, 13], [7, 14],
  [8, 11], [8, 12], [8, 13], [8, 14], [9, 11], [9, 12], [9, 13], [9, 14], [10, 11], [10, 12], [10, 13], [10, 14],
  [0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2],
  [4, 5], [4, 6], [5, 4], [5, 6], [6, 4], [6, 5], [7, 8], [7, 9], [7, 10], [8, 7], [8, 9], [8, 10],
  [9, 7], [9, 8], [9, 10], [10, 7], [10, 8], [10, 9], [11, 12], [11, 13], [11, 14], [12, 11], [12, 13], [12, 14],
  [13, 11], [13, 12], [13, 14], [14, 11], [14, 12], [14, 13],
]

// Sivas Merkez: 16 node. TSO(9) ve EVCİLER(3) gerçek API ID'leri;
// 4 YOL/MİLLET BAHÇESİ/KARAYOLLARI GAN-sentetik olduğu için gerçek sensör ID'si yok,
// bilinçli olarak ayrılmış dummy ID atandı (90001-90003).
export const SIVAS_MERKEZ_NODE_MAP: NodeMap = {
  9:     { A: 0, B: 1,       D: 2 },        // TSO (C yok)
  90001: { A: 3, B: 4, C: 5, D: 6 },        // 4 YOL (dummy id)
  90002: { A: 7, B: 8, C: 9 },              // MİLLET BAHÇESİ (dummy id, D yok)
  3:     { A: 10, B: 11 },                  // EVCİLER (C, D yok)
  90003: { A: 12, B: 13, C: 14, D: 15 },    // KARAYOLLARI (dummy id)
}

// data/Sivas/directed_sivas.csv içeriği (from,to) — 16 node, tam bağlı graf
export const SIVAS_MERKEZ_GRAPH_EDGES: GraphEdges = (() => {
  const edges: GraphEdges = []
  for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
      if (i !== j) edges.push([i, j])
    }
  }
  return edges
})()

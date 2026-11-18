package com.apkcreator.offlineart

import java.io.File
import org.json.JSONObject

class ClipTokenizerKt(private val vocab: Map<String, Int>, private val ranks: Map<String, Int>) {
    data class Enc(val ids: IntArray, val mask: IntArray)

    fun encode(text: String): Enc {
        val low = StringBuilder(text.length)
        for (ch in text) low.append(if (ch in 'A'..'Z') ch + ('a' - 'A') else ch)
        val t = low.toString()
        val re = Regex("[ ]?'(?:s|t|re|ve|m|ll|d)| ?\\p{L}+| ?[0-9]+| ?[^\\s\\p{L}0-9]+|\\s+")
        val ids = ArrayList<Int>()
        for (mm in re.findAll(t)) {
            var tok = mm.value
            if (tok.isBlank()) continue
            if (tok[0].code == 32) tok = tok.substring(1)
            if (tok.isEmpty()) continue
            val bytes = tok.toByteArray(Charsets.UTF_8)
            val units = ArrayList<String>(bytes.size + 1)
            for (b in bytes) units.add(BYTE_MAP[b.toInt() and 0xFF].toString())
            units[units.size - 1] = units.last() + "</w>"
            val pieces = bpe(units)
            for ((i, p) in pieces.withIndex()) {
                if (i == pieces.size - 1 && p.endsWith("</w>") && p.dropLast(4).any { it.code > 127 }) { ids.add(0); continue }
                ids.add(vocab[p] ?: 0)
            }
        }
        val full = IntArray(77) { 49407 }
        val m2 = IntArray(77) { 0 }
        full[0] = 49406; m2[0] = 1
        val n = minOf(ids.size, 75)
        for (i in 0 until n) { full[1 + i] = ids[i]; m2[1 + i] = 1 }
        full[1 + n] = 49407; m2[1 + n] = 1
        return Enc(full, m2)
    }

    private fun bpe(units: List<String>): List<String> {
        val w = units.toMutableList()
        while (w.size >= 2) {
            var bi = -1; var br = Int.MAX_VALUE
            for (i in 0 until w.size - 1) {
                val a = w[i]; val b = w[i + 1]
                if (a.any { it.code > 127 } && b.endsWith("</w>")) continue
                val r = ranks[a + 0x01.toChar() + b] ?: continue
                if (r < br) { br = r; bi = i }
            }
            if (bi < 0) break
            w[bi] = w[bi] + w[bi + 1]
            w.removeAt(bi + 1)
        }
        return w
    }

    companion object {
        private val BYTE_MAP: CharArray by lazy {
            val bs = ArrayList<Int>()
            for (i in 33..126) bs.add(i)
            for (i in 161..172) bs.add(i)
            for (i in 174..255) bs.add(i)
            val cs = ArrayList(bs)
            var n = 0
            for (b in 0..255) if (!bs.contains(b)) { bs.add(b); cs.add(256 + n); n++ }
            val m = CharArray(256)
            for (i in bs.indices) m[bs[i]] = cs[i].toChar()
            m
        }

        fun load(modelsDir: File): ClipTokenizerKt {
            val vocab = HashMap<String, Int>()
            val jo = JSONObject(File(modelsDir, "tokenizer/vocab.json").readText())
            val keys = jo.keys()
            while (keys.hasNext()) { val k = keys.next(); vocab[k] = jo.getInt(k) }
            val ranks = HashMap<String, Int>()
            File(modelsDir, "tokenizer/merges.txt").readLines().drop(1).forEachIndexed { i, ln ->
                if (ln.isEmpty()) return@forEachIndexed
                val sp = ln.indexOf(' ')
                if (sp < 0) return@forEachIndexed
                ranks[ln.substring(0, sp) + 0x01.toChar() + ln.substring(sp + 1)] = i
            }
            return ClipTokenizerKt(vocab, ranks)
        }
    }
}
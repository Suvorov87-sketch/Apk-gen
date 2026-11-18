package com.apkcreator.offlineart

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Random
import kotlin.math.max
import kotlin.math.min

class DiffusionPipelineKt(modelsDir: File, threads: Int) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val tok = ClipTokenizerKt.load(modelsDir)
    private val sched = LcmSchedulerKt()
    private val te: OrtSession
    private val unet: OrtSession
    private val vae: OrtSession

    init {
        val o = OrtSession.SessionOptions()
        o.setIntraOpNumThreads(max(1, threads))
        te = env.createSession(File(modelsDir, "text_encoder/model.onnx").absolutePath, o)
        unet = env.createSession(File(modelsDir, "unet/model.onnx").absolutePath, o)
        vae = env.createSession(File(modelsDir, "vae_decoder/model.onnx").absolutePath, o)
        require(te.outputNames.size == 1) { "TE: unexpected outputs" }
        require(unet.outputNames.size == 1) { "UNet: unexpected outputs" }
        require(vae.outputNames.size == 1) { "VAE: unexpected outputs" }
    }

    private fun pickInput(s: OrtSession, vararg names: String, wantFloat4D: Boolean = false): String {
        val have = s.inputNames
        for (n in names) if (have.contains(n)) return n
        for (n in have) {
            try {
                val info = s.inputInfo[n]!!.info
                if (info is ai.onnxruntime.TensorInfo && info.shape.size == 4 && wantFloat4D) return n
            } catch (e: Exception) {}
        }
        return have.first()
    }

    private fun firstFloat4D(s: OrtSession, batchDim: Int, ch: Int): String {
        for (n in arrayOf("out_sample", "sample")) if (s.outputNames.contains(n)) return n
        for (n in s.outputNames) {
            try {
                val info = s.outputInfo[n]!!.info
                if (info is ai.onnxruntime.TensorInfo && info.type == OnnxJavaType.FLOAT) {
                    val sh = info.shape
                    if (sh.size == 4 && (sh[1] == ch.toLong() || sh[1] < 0)) return n
                }
            } catch (e: Exception) {}
        }
        return s.outputNames.first()
    }

    private fun encode(text: String): FloatArray {
        val e = tok.encode(text)
        val ids = LongArray(77) { e.ids[it].toLong() }
        val inName = if (te.inputNames.contains("input_ids")) "input_ids" else te.inputNames.first()
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, 77)).use { t ->
            te.run(mapOf(inName to t)).use { r ->
                val v = r[0].value as Array<Array<FloatArray>>
                val out = FloatArray(77 * 768)
                for (i in 0 until 77) for (j in 0 until 768) out[i * 768 + j] = v[0][i][j]
                return out
            }
        }
    }

    private fun randn(n: Int, seed: Long): FloatArray {
        val r = Random(seed)
        val a = FloatArray(n)
        for (i in 0 until n) a[i] = r.nextGaussian().toFloat()
        return a
    }

    fun generate(prompt: String, neg: String, seed: Long, steps: Int, guidance: Float, size: Int, onStep: (Int, Int) -> Unit): Bitmap {
        val h = size / 8; val w = size / 8
        val cond = encode(prompt)
        val useCfg = guidance != 1.0f
        val uncond = if (useCfg) encode(if (neg.isBlank()) "" else neg) else null
        var lat = randn(4 * h * w, seed)
        val ts = sched.timesteps(steps)
        val sampleName = pickInput(unet, "sample", wantFloat4D = true)
        val timeName = if (unet.inputNames.contains("timestep")) "timestep" else unet.inputNames.first { it != sampleName }
        var embedName = "encoder_hidden_states"
        if (!unet.inputNames.contains(embedName)) {
            embedName = unet.inputNames.first { it != sampleName && it != timeName }
        }
        val outName = firstFloat4D(unet, 1, 4)
        for (si in ts.indices) {
            val t = ts[si].toInt()
            val prev = if (si + 1 < ts.size) ts[si + 1].toInt() else -1
            val latIn: FloatArray
            val embIn: FloatArray
            val batch: Int
            if (useCfg) {
                batch = 2
                latIn = lat + lat
                embIn = uncond!! + cond
            } else { batch = 1; latIn = lat; embIn = cond }
            val ch = 4 * h * w
            OnnxTensor.createTensor(env, FloatBuffer.wrap(latIn), longArrayOf(batch.toLong(), 4, h.toLong(), w.toLong())).use { sT ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(t.toLong())), longArrayOf(1)).use { tT ->
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(embIn), longArrayOf(batch.toLong(), 77, 768)).use { eT ->
                        unet.run(mapOf(sampleName to sT, timeName to tT, embedName to eT)).use { r ->
                            val raw = (r[0].value) as Array<Array<Array<FloatArray>>>
                            val pred: FloatArray
                            if (useCfg) {
                                val split = raw.size / 2
                                val flat = flatten4(raw)
                                val u = flat.copyOfRange(0, ch)
                                val c = flat.copyOfRange(ch, ch * 2)
                                pred = FloatArray(ch) { i -> u[i] + guidance * (c[i] - u[i]) }
                            } else { pred = flatten4(raw) }
                            lat = sched.step(lat, pred, t, prev)
                        }
                    }
                }
            }
            onStep(si + 1, ts.size)
        }
        for (i in lat.indices) lat[i] = (lat[i] / 0.18215f)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(lat), longArrayOf(1, 4, h.toLong(), w.toLong())).use { lT ->
            val vin = if (vae.inputNames.contains("latent_sample")) "latent_sample" else vae.inputNames.first()
            vae.run(mapOf(vin to lT)).use { r ->
                val vout = firstFloat4D(vae, 1, 3)
                val img = (r[0].value) as Array<Array<Array<FloatArray>>>
                return toBitmap(img, size)
            }
        }
    }

    private fun flatten4(a: Array<Array<Array<FloatArray>>>): FloatArray {
        val b = a.size; val c = a[0].size; val h = a[0][0].size; val w = a[0][0][0].size
        val out = FloatArray(b * c * h * w)
        var k = 0
        for (bb in 0 until b) for (cc in 0 until c) for (yy in 0 until h) for (xx in 0 until w) out[k++] = a[bb][cc][yy][xx]
        return out
    }

    private fun toBitmap(img: Array<Array<Array<FloatArray>>>, size: Int): Bitmap {
        val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val px = IntArray(size * size)
        for (y in 0 until size) for (x in 0 until size) {
            fun ch(c: Int): Int {
                val v = img[0][c][y][x] * 0.5f + 0.5f
                return (min(1f, max(0f, v)) * 255).toInt()
            }
            px[y * size + x] = (255 shl 24) or (ch(0) shl 16) or (ch(1) shl 8) or ch(2)
        }
        bmp.setPixels(px, 0, size, 0, 0, size, size)
        return bmp
    }

    override fun close() {
        try { te.close() } catch (e: Exception) {}
        try { unet.close() } catch (e: Exception) {}
        try { vae.close() } catch (e: Exception) {}
        try { env.close() } catch (e: Exception) {}
    }
}
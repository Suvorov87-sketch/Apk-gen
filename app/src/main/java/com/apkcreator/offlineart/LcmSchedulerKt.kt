package com.apkcreator.offlineart

import kotlin.math.round
import kotlin.math.sqrt

class LcmSchedulerKt(
    betaStart: Double = 0.00085,
    betaEnd: Double = 0.012,
    trainSteps: Int = 1000,
    private val stepsOffset: Int = 1
) {
    private val acp: DoubleArray
    init {
        acp = DoubleArray(trainSteps)
        var acc = 1.0
        for (i in 0 until trainSteps) {
            val t = if (trainSteps == 1) 0.0 else i.toDouble() / (trainSteps - 1)
            val b = sqrt(betaStart) + (sqrt(betaEnd) - sqrt(betaStart)) * t
            acc *= 1.0 - b * b
            acp[i] = acc
        }
    }

    fun timesteps(n: Int): LongArray {
        val ratio = trainSteps / n
        val ts = (0 until n).map { round(it * ratio.toDouble()).toLong() }.reversed()
        return ts.map { it + stepsOffset }.toLongArray()
    }

    fun step(sample: FloatArray, pred: FloatArray, t: Int, prevT: Int): FloatArray {
        val aT = acp[t]
        val aP = if (prevT < 0) 1.0 else acp[prevT]
        val bT = 1.0 - aT
        val bP = 1.0 - aP
        val sT = sqrt(aT); val sbT = sqrt(bT)
        val sP = sqrt(aP); val sbP = sqrt(bP)
        val out = FloatArray(sample.size)
        for (i in sample.indices) {
            val x0 = (sample[i] - sbT * pred[i]) / sT
            out[i] = (sP * x0 + sbP * pred[i]).toFloat()
        }
        return out
    }
}
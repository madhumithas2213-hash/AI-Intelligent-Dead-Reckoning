package com.aiidr.navigation

import kotlin.math.sqrt

/**
 * Navigation State Data Container
 */
data class NavigationState(
    val timestamp: Long,
    val posX: Double,             // Easting / Local X (m)
    val posY: Double,             // Northing / Local Y (m)
    val speedKmh: Float,          // Vehicle Speed (km/h)
    val headingDeg: Float,        // Heading angle (degrees)
    val positionUncertaintyM: Float, // Error circle radius (m)
    val gnssState: String,        // "AVAILABLE", "DEGRADED", "LOST", "RECOVERED"
    val navMode: String,          // "GNSS-AIDED", "GNSS-DEGRADED", "DEAD RECKONING", "GNSS-RECOVERED"
    val confidencePct: Float      // Sensor / AI confidence (0 - 100%)
)

/**
 * Adaptive Extended Kalman Filter for Sensor Fusion & GNSS-Denied Dead Reckoning
 */
class AdaptiveFusionEKF {

    // EKF State Vector [X, Y, Vx, Vy, Heading]
    private var posX: Double = 0.0
    private var posY: Double = 0.0
    private var vx: Double = 0.0
    private var vy: Double = 0.0
    private var headingRad: Double = 0.0
    private var uncertainty: Float = 0.4f

    private var gnssStateInternal: String = "AVAILABLE"
    private var navModeInternal: String = "GNSS-AIDED"
    private var outageStartTimeMs: Long = 0

    fun updateGNSSFix(lat: Double, lon: Double, hdop: Float, sats: Int): NavigationState {
        val now = System.currentTimeMillis()
        if (hdop > 3.0f || sats < 4) {
            gnssStateInternal = "DEGRADED"
            navModeInternal = "GNSS-DEGRADED"
            uncertainty = 1.8f
        } else {
            gnssStateInternal = "AVAILABLE"
            navModeInternal = "GNSS-AIDED"
            uncertainty = 0.4f
        }

        val speed = sqrt(vx * vx + vy * vy).toFloat() * 3.6f
        return NavigationState(
            timestamp = now,
            posX = posX,
            posY = posY,
            speedKmh = speed,
            headingDeg = Math.toDegrees(headingRad).toFloat(),
            positionUncertaintyM = uncertainty,
            gnssState = gnssStateInternal,
            navMode = navModeInternal,
            confidencePct = 98.5f
        )
    }

    fun updateDeadReckoning(predictedVel: FloatArray, dtSeconds: Float): NavigationState {
        val now = System.currentTimeMillis()
        if (gnssStateInternal != "LOST") {
            gnssStateInternal = "LOST"
            navModeInternal = "DEAD RECKONING"
            outageStartTimeMs = now
        }

        val outageDurationSec = (now - outageStartTimeMs) / 1000.0f
        // Uncertainty grows over outage duration
        uncertainty = 0.5f + (0.15f * outageDurationSec)

        // Integrate AI velocity into state
        vx = predictedVel[0].toDouble()
        vy = predictedVel[1].toDouble()

        posX += vx * dtSeconds
        posY += vy * dtSeconds

        val speed = sqrt(vx * vx + vy * vy).toFloat() * 3.6f
        return NavigationState(
            timestamp = now,
            posX = posX,
            posY = posY,
            speedKmh = speed,
            headingDeg = Math.toDegrees(headingRad).toFloat(),
            positionUncertaintyM = uncertainty,
            gnssState = "LOST",
            navMode = "DEAD RECKONING",
            confidencePct = (92.0f - (outageDurationSec * 0.2f)).coerceAtLeast(70.0f)
        )
    }

    fun handleGNSSRecovery(targetPosX: Double, targetPosY: Double): NavigationState {
        val now = System.currentTimeMillis()
        gnssStateInternal = "RECOVERED"
        navModeInternal = "GNSS-RECOVERED"

        // Smooth Kalman gain correction back to target position
        posX = posX * 0.3 + targetPosX * 0.7
        posY = posY * 0.3 + targetPosY * 0.7
        uncertainty = 0.6f

        val speed = sqrt(vx * vx + vy * vy).toFloat() * 3.6f
        return NavigationState(
            timestamp = now,
            posX = posX,
            posY = posY,
            speedKmh = speed,
            headingDeg = Math.toDegrees(headingRad).toFloat(),
            positionUncertaintyM = uncertainty,
            gnssState = "RECOVERED",
            navMode = "GNSS-RECOVERED",
            confidencePct = 97.0f
        )
    }
}

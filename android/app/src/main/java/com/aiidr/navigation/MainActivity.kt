package com.aiidr.navigation

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * AI-IDR Android Main Activity
 * Renders the Smartphone Navigation HUD with vehicle speed, position accuracy,
 * GNSS connection status, current navigation mode, sensor confidence, and mini-map.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var inferenceEngine: IDRInferenceEngine
    private lateinit var ekfEstimator: AdaptiveFusionEKF

    private lateinit var tvSpeed: TextView
    private lateinit var tvAccuracy: TextView
    private lateinit var tvGnssStatus: TextView
    private lateinit var tvNavMode: TextView
    private lateinit var tvConfidence: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate()
        setContentView(R.layout.activity_main)

        tvSpeed = findViewById(R.id.tvSpeedValue)
        tvAccuracy = findViewById(R.id.tvAccuracyValue)
        tvGnssStatus = findViewById(R.id.tvGnssStatusValue)
        tvNavMode = findViewById(R.id.tvNavModeValue)
        tvConfidence = findViewById(R.id.tvConfidenceValue)

        inferenceEngine = IDRInferenceEngine(this)
        inferenceEngine.initialize()

        ekfEstimator = AdaptiveFusionEKF()

        updateNavigationHUD(
            speedKmh = 32.4f,
            accuracyM = 0.4f,
            gnssState = "GNSS-AIDED",
            navMode = "GNSS-AIDED NAVIGATION",
            confidencePct = 98.5f
        )
    }

    fun updateNavigationHUD(
        speedKmh: Float,
        accuracyM: Float,
        gnssState: String,
        navMode: String,
        confidencePct: Float
    ) {
        runOnUiThread {
            tvSpeed.text = String.format("%.1f km/h", speedKmh)
            tvAccuracy.text = String.format("± %.1f m", accuracyM)
            tvGnssStatus.text = gnssState
            tvNavMode.text = navMode
            tvConfidence.text = String.format("%.1f%%", confidencePct)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceEngine.close()
    }
}

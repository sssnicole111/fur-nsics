import { useState } from 'react'
import './ResultScreen.css'

export default function ResultScreen({ result, onScanAgain, onHome }) {
  const [showHeatmap, setShowHeatmap] = useState(true)
  
  const isPositive = result?.label === 'POSITIVE'
  const isAttention = result?.displayLabel === 'NEEDS ATTENTION'
  const confidence = Math.round((result?.confidence || 0) * 100)

  let actionText = 'Continue regular monitoring'
  let actionColor = '#1E8449'
  if (result?.displayLabel === 'RINGWORM SUSPECTED') {
    actionText = 'Urgent veterinary referral required immediately'
    actionColor = '#C0392B'
  } else if (result?.displayLabel === 'NEEDS ATTENTION') {
    actionText = 'Veterinary consultation recommended within 24 hours'
    actionColor = '#E67E22'
  }

  const headerColor = isPositive
    ? (isAttention ? 'var(--warning)' : 'var(--danger)')
    : 'var(--success)'

  // LMS color based on category
  const lmsColor = result?.lms === 'Classic Ring Pattern' ? '#C0392B'
    : result?.lms === 'Partial Ring Pattern' ? '#E67E22'
    : '#7F8C8D'

  const lmsDescription = result?.lms === 'Classic Ring Pattern'
    ? 'Activation shape strongly matches dermatophyte ring morphology — high-confidence ringworm.'
    : result?.lms === 'Partial Ring Pattern'
    ? 'Partial circular pattern detected — possible early or healing ringworm. Vet consultation recommended.'
    : 'Shape does not match ring pattern — atypical or non-ringworm condition. Vet verification required.'

  return (
    <div className="result-screen">

      {/* Top Bar */}
      <div className="top-bar">
        <div className="top-bar-left">
          <span className="logo-text">furnsics</span>
        </div>
        <div className="top-bar-right">
          {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
        </div>
      </div>

      <div className="result-body">

        {/* Hero Image Card */}
        {result?.heatmapImage && (
          <div className="result-section hero-image-card">
            <img
              src={showHeatmap ? result.heatmapImage : result.rawImage}
              alt="Diagnosis view"
              className="heatmap-img-full"
            />
            {showHeatmap && (
              <div className="hero-overlay-bottom">
                 <span className="target-icon">◎</span> Red areas indicate regions of influence
              </div>
            )}
            <div className="hero-switch-overlay">
              <label className="switch">
                <input 
                  type="checkbox" 
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
          </div>
        )}

        {/* Result Banner (No Container) */}
        <div className="result-banner">
           <h1 className="result-label" style={{ color: (isPositive && !isAttention) ? '#4A0E1B' : 'var(--ink)' }}>{result?.displayLabel}</h1>
           <p className="result-confidence" style={{ color: 'var(--ink-2)' }}>
             AI confidence: {confidence}%
           </p>
        </div>

        {/* Stats Row */}
        {result?.lms && result.lms !== 'N/A' && (
          <div className="stats-row">
            <div className="result-section stat-card">
              <div className="stat-label">LESION PATTERN</div>
              <div className="stat-value">{result.lms}</div>
            </div>
            <div className="result-section stat-card">
              <div className="stat-label">CIRCULARITY</div>
              <div className="stat-value">{result.circularity}</div>
            </div>
          </div>
        )}

        {/* Description Card */}
        {result?.lms && result.lms !== 'N/A' && (
          <div className="result-section desc-card">
            <p className="desc-text">{lmsDescription}</p>
          </div>
        )}

        {/* Action Card */}
        <div className="result-section action-card">
           <svg className="action-icon" style={{ color: actionColor }} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
             <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
             <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
           </svg>
           <h2 className="action-title" style={{ color: 'var(--ink)' }}>{actionText}</h2>
           <p className="action-subtitle" style={{ color: 'var(--ink-2)' }}>
             {isPositive 
                ? 'Preliminary screening result only. Veterinarian confirmation required.'
                : 'No ringworm indicators detected. Monitor and consult a vet if symptoms develop.'}
           </p>
        </div>

        {/* Disclaimer */}
        <div className="result-disclaimer">
          Fur-Scan is a triage aid only. Not a veterinary diagnosis.
        </div>

        {/* Action buttons */}
        <div className="result-actions" style={{ flexDirection: 'row' }}>
          <button className="btn-secondary outline" style={{ flex: 1 }} onClick={onHome}>
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg> Back to home
          </button>
          <button className="btn-secondary outline" style={{ flex: 1 }} onClick={onScanAgain}>
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg> Scan again
          </button>
        </div>

      </div>
    </div>
  )
}

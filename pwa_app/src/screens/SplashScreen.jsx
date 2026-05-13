import { useEffect, useState } from 'react'
import './SplashScreen.css'

export default function SplashScreen({ onStart }) {
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 100)
    return () => clearTimeout(t)
  }, [])

  const features = [
    { 
      label: 'AI Detection', 
      desc: 'MobileNetV3 on-device',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
          <polyline points="7.5 4.21 12 6.81 16.5 4.21"></polyline>
          <polyline points="7.5 19.79 7.5 14.6 3 12"></polyline>
          <polyline points="21 12 16.5 14.6 16.5 19.79"></polyline>
          <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
          <line x1="12" y1="22.08" x2="12" y2="12"></line>
        </svg>
      )
    },
    { 
      label: 'Grad-CAM Heatmaps', 
      desc: 'Visual evidence per result',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
          <circle cx="8.5" cy="8.5" r="1.5"></circle>
          <polyline points="21 15 16 10 5 21"></polyline>
        </svg>
      )
    },
    { 
      label: 'Lesion Shape Classifier', 
      desc: 'Circularity-based ring pattern analysis',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"></path>
          <path d="M2 12h20"></path>
        </svg>
      )
    },
    { 
      label: 'Works Offline', 
      desc: 'No internet required',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M5 12.55a11 11 0 0 1 14.08 0"></path>
          <path d="M1.42 9a16 16 0 0 1 21.16 0"></path>
          <path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path>
          <line x1="12" y1="20" x2="12.01" y2="20"></line>
        </svg>
      )
    },
  ]

  return (
    <div className={`splash ${ready ? 'ready' : ''}`}>

      <div className="splash-top">
        <div className="splash-badge">Edge-AI · Offline-First</div>
      </div>

      <div className="splash-hero">
        <div className="logo-mark">
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
            {/* Circular ring shape representing dermatophyte lesion */}
            <circle cx="18" cy="18" r="14" stroke="white" strokeWidth="2.5" fill="none"/>
            <circle cx="18" cy="18" r="8" stroke="white" strokeWidth="1.5" fill="none" strokeDasharray="3 2"/>
            <circle cx="18" cy="18" r="3" fill="white"/>
          </svg>
        </div>
        <h1 className="splash-title">Fur-nsics</h1>
        <p className="splash-sub">Canine Ringworm Screening</p>
      </div>

      <div className="splash-features">
        {features.map((f) => (
          <div className="feature-row" key={f.label}>
            <div className="feature-icon-wrap">
              {f.icon}
            </div>
            <div>
              <div className="feature-label">{f.label}</div>
              <div className="feature-desc">{f.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="splash-footer">
        <button className="btn-primary" onClick={onStart}>
          Begin Screening
        </button>
        <p className="splash-disclaimer">Triage use only · Not a veterinary diagnosis</p>
      </div>

    </div>
  )
}

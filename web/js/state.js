export const ENROLL_PHASES = ['front', 'left', 'right'];
export const ENROLL_REQUIRED_GOOD_FRAMES = 2;
export const ENROLL_CAPTURE_INTERVAL_MS = 700;
export const ENROLL_PHASE_META = {
    front: {
        label: 'FRONT',
        instruction: 'Look straight into the camera',
        hint: 'Keep your face centered and hold still',
        arrow: 'arrow_upward',
        guideClass: 'front'
    },
    left: {
        label: 'LEFT',
        instruction: 'Turn your face toward the LEFT arrow',
        hint: 'Turn slowly and hold the pose until 2 frames pass',
        arrow: 'arrow_back',
        guideClass: 'left'
    },
    right: {
        label: 'RIGHT',
        instruction: 'Turn your face toward the RIGHT arrow',
        hint: 'Turn slowly and hold the pose until 2 frames pass',
        arrow: 'arrow_forward',
        guideClass: 'right'
    }
};

export const state = {
    currentTheme: localStorage.getItem('theme') || 'light',
    scanModePreference: localStorage.getItem('scanMode') || 'auto',
    effectiveScanMode: null,
    scanCapabilities: null,
    isAutoScanning: false,
    autoScanInterval: null,
    scanIntervalMs: 2500,
    scanCount: 0,
    presentSet: new Set(),
    presentStudents: new Map(),
    activeChallenge: null,
    isChallengeRunning: false,
    sessionActive: false,
    sessionStats: { total: 0, present: 0, absent: 0 },
    currentView: 'active',
    cameraSource: 'browser',
    localStream: null,
    demoPlaying: false,
    enrollState: createEnrollState()
};

export function createEnrollState(overrides = {}) {
    return {
        id: null,
        name: null,
        className: null,
        stream: null,
        currentPhase: 0,
        autoTimer: null,
        isValidating: false,
        captures: { front: [], left: [], right: [] },
        ...overrides
    };
}

export function setEnrollState(nextState) {
    state.enrollState = nextState;
}

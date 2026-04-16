import { state } from './state.js';
import { closeModal, setupTabs, showModal, toggleTheme, updateClock } from './ui.js';
import { checkSessionStatus, hideSummary, uiEndSession, uiStartSession } from './session.js';
import { archiveStudent, loadStudents, openEditModal, restoreStudent, saveStudentEdit, setStudentView, showStudentDetail } from './students.js';
import { captureEnrollFrame, closeEnrollCameraModal, proceedToCameraEnroll, proceedToManualEnroll, retakeAngle, showStudentCreateModal, startReEnroll, submitEnroll, validateManualFile } from './enrollment.js';
import { hideDetail, loadHistory, showHistoryDetail } from './history.js';
import { setCameraSource, skipAttendanceSuccessOverlay, testAttendanceSuccessOverlay, toggleDemoVideo, updateAutoScanInterval } from './scan.js';

function exposeGlobals() {
    Object.assign(window, {
        toggleTheme,
        showModal,
        closeModal,
        uiStartSession,
        uiEndSession,
        hideSummary,
        loadStudents,
        setStudentView,
        archiveStudent,
        restoreStudent,
        showStudentDetail,
        openEditModal,
        saveStudentEdit,
        showStudentCreateModal,
        startReEnroll,
        proceedToCameraEnroll,
        proceedToManualEnroll,
        closeEnrollCameraModal,
        retakeAngle,
        captureEnrollFrame,
        validateManualFile,
        submitEnroll,
        hideDetail,
        showHistoryDetail,
        setCameraSource,
        skipAttendanceSuccessOverlay,
        testAttendanceSuccessOverlay,
        updateAutoScanInterval,
        toggleDemoVideo
    });
}

document.addEventListener('DOMContentLoaded', () => {
    exposeGlobals();
    document.documentElement.setAttribute('data-theme', state.currentTheme);
    setupTabs();
    updateClock();
    setInterval(updateClock, 1000);
    checkSessionStatus();
    loadStudents('active');
    loadHistory();
});

/**
 * Shared utility functions for Fitness Overlays
 * Handles unit conversions and formatting based on user preference.
 */

// Get preference from window object, default to 'meters' (Metric)
function getMeasurementPreference() {
  return window.measurementPreference || 'meters';
}

function isImperial() {
  return getMeasurementPreference() === 'feet';
}

function formatDuration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${remainingSeconds}s`;
  }
  return `${minutes}m ${remainingSeconds}s`;
}

function formatDistance(meters) {
  if (isImperial()) {
    // Convert to miles
    const miles = meters * 0.000621371;
    return miles.toFixed(2) + ' mi';
  } else {
    // Metric (km)
    return (meters / 1000).toFixed(2) + ' km';
  }
}

function formatPace(seconds, distance) {
  if (!distance) return "0:00";

  let paceInSeconds;
  let unit;
  if (isImperial()) {
    // Minutes per mile
    const miles = distance * 0.000621371;
    paceInSeconds = seconds / miles;
    unit = '/mi';
  } else {
    // Minutes per km
    paceInSeconds = seconds / (distance / 1000);
    unit = '/km';
  }

  const minutes = Math.floor(paceInSeconds / 60);
  const remainingSeconds = Math.ceil(paceInSeconds % 60);
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')} ${unit}`;
}

function formatSpeed(distance, timeInSeconds) {
  if (!distance || !timeInSeconds) return "0.0";

  if (isImperial()) {
    // mph
    const miles = distance * 0.000621371;
    const hours = timeInSeconds / 3600;
    const mph = miles / hours;
    return `${mph.toFixed(1)} mph`;
  } else {
    // km/h
    const km = distance / 1000;
    const hours = timeInSeconds / 3600;
    const kmh = km / hours;
    return `${kmh.toFixed(1)} km/h`;
  }
}

function formatElevation(meters) {
  if (isImperial()) {
    // feet
    const feet = meters * 3.28084;
    return `${Math.round(feet)} ft`;
  } else {
    // meters
    return `${Math.round(meters)} m`;
  }
}

function formatSwimPace(seconds, distance) {
  if (!distance || distance < 100) return 'N/A';

  if (isImperial()) {
    // Pace per 100 yards
    const yards = distance * 1.09361;
    const pacePer100Yards = seconds / (yards / 100);
    const minutes = Math.floor(pacePer100Yards / 60);
    const secs = Math.round(pacePer100Yards % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')} /100yd`;
  } else {
    // Pace per 100 meters
    const pacePer100m = seconds / (distance / 100);
    const minutes = Math.floor(pacePer100m / 60);
    const secs = Math.round(pacePer100m % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')} /100m`;
  }
}

// Export functions to global scope
window.formatDuration = formatDuration;
window.formatDistance = formatDistance;
window.formatPace = formatPace;
window.formatSpeed = formatSpeed;
window.formatElevation = formatElevation;
window.formatSwimPace = formatSwimPace;

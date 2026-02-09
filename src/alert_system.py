"""
Simple alerting utilities — can be wrapped by a web service or CLI.
"""
def format_alert(sample_id, lap_index, probability, threshold, horizon_laps=12):
    if probability >= threshold:
        return {
            'sample_id': int(sample_id),
            'lap': int(lap_index),
            'alert': True,
            'message': f"Tire wear predicted within next {horizon_laps} laps (p={probability:.3f})"
        }
    else:
        return {
            'sample_id': int(sample_id),
            'lap': int(lap_index),
            'alert': False,
            'message': f"Low wear risk (p={probability:.3f})"
        }

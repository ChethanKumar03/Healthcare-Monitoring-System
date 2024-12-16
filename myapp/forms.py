from django import forms

class HealthInfoForm(forms.ModelForm):
    class Meta:
        model = HealthInfo
        fields = ['phone_number', 'age', 'gender', 'heart_rate', 'low_bp', 'high_bp', 'height', 'weight', 'body_temperature','prediction_result']
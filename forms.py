from django import forms


class UploadFileForm(forms.Form):
    file = forms.FileField()


class DeductionStatsForm(forms.Form):
    start_month = forms.DateField(
        label='开始月份',
        widget=forms.DateInput(attrs={'type': 'month'}),
        required=True
    )
    end_month = forms.DateField(
        label='结束月份',
        widget=forms.DateInput(attrs={'type': 'month'}),
        required=True
    )
    assessment_item = forms.MultipleChoiceField(
        label='选择考核大项',
        widget=forms.SelectMultiple,
        choices=[]  # 实际的选项需要在视图中动态加载
    )
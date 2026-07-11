{{- define "heart-api.name" -}}
heart-api
{{- end -}}

{{- define "heart-api.labels" -}}
app: {{ include "heart-api.name" . }}
chart: {{ .Chart.Name }}-{{ .Chart.Version }}
release: {{ .Release.Name }}
{{- end -}}

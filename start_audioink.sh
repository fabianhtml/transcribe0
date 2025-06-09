#!/bin/bash

# Script para iniciar AudioInk en segundo plano
echo "🚀 Iniciando AudioInk..."

# Verificar si ya está corriendo
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ AudioInk ya está corriendo en el puerto 8501"
    echo "🌐 Abre: http://localhost:8501"
else
    echo "📱 Iniciando AudioInk en segundo plano..."
    nohup streamlit run audioink.py --server.headless true --server.port 8501 > audioink.log 2>&1 &
    echo "✅ AudioInk iniciado en segundo plano"
    echo "🌐 Abre: http://localhost:8501"
    echo "📄 Log: audioink.log"
fi
# Mubser — AI Assistive System for Visually Impaired Pilgrims

Mubser is an AI-powered assistive system designed to support visually impaired pilgrims during Hajj and Umrah.

The system combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), computer vision, offline voice processing, and edge AI into an integrated assistive experience.

> Extended Hackathon Project

---

## Overview

Mubser aims to reduce the need for manual assistance by providing contextual guidance and AI-powered interaction for visually impaired pilgrims.

The system is designed around a hybrid AI architecture that combines:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Computer Vision
- Offline Voice Recognition
- Arabic Text-to-Speech
- Edge AI
- Mobile Application
- Smart-glasses integration
- Spatial and navigation assistance

---

## Key Features

### AI Assistant
Provides contextual responses using LLM-based reasoning.

### Arabic RAG System
Retrieves relevant information from a structured Arabic knowledge base before generating responses.

### Hybrid LLM Architecture
Uses Ollama for local LLM execution with Gemini API as a cloud fallback.

### Computer Vision
Uses YOLOv8 and OpenCV for visual understanding, including:

- Landmark recognition
- Obstacle detection
- Environment understanding
- Tawaf counting

### Voice Interaction

The pipeline supports:

Voice Input → LLM Reasoning → RAG Retrieval → Computer Vision → Arabic Voice Output

The architecture is designed to minimize manual interaction.

---

## System Architecture

```text
User Voice Input
       ↓
Offline Voice Recognition
       ↓
LLM Reasoning
       ↓
RAG Retrieval
       ↓
Contextual Response
       ↓
Computer Vision
       ↓
Arabic Text-to-Speech
       ↓
User

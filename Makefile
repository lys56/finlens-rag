.PHONY: help install test run clean docker

help:
	@echo "Financial RAG - Makefile 命令"
	@echo ""
	@echo "make install    - 安装依赖"
	@echo "make test       - 快速测试"
	@echo "make run        - 启动 Web 应用"
	@echo "make demo       - 运行命令行演示"
	@echo "make pipeline   - 运行完整流水线"
	@echo "make clean      - 清理生成的文件"
	@echo "make docker     - 构建 Docker 镜像"
	@echo ""

install:
	pip install -r requirements.txt
	@echo "依赖安装完成"

test:
	python run_pipeline.py --test
	@echo "测试完成"

run:
	streamlit run app.py

demo:
	python demo.py

pipeline:
	python run_pipeline.py --tickers AAPL MSFT --amount 2

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "清理完成"

docker:
	docker build -t financial-rag:latest .
	@echo "Docker 镜像构建完成"

docker-run:
	docker-compose up -d
	@echo "Docker 容器已启动"

docker-stop:
	docker-compose down
	@echo "Docker 容器已停止"

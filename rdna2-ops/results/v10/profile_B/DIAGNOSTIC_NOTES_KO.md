# 첫 운영 프로파일은 조건 복구용 진단 자료

3 epoch/93 step, step40–44 trace, KFD max7 queues/6 clients(워커4개는0queue), 보호 SHA 동일이다. epoch3은12.5467초다. 다음 이유로 최종 B 기준선에는 사용하지 않는다.

1. config/filelist만 복사한 새 모델에는 model_info.json이 없다. 운영 train.py는 이를 109speakers/contentvec로 처리하지만 원본 voice_model_A metadata는1speaker/KLM-SVM이다. 학습 모델 초기값과 reference 경로가 달라졌다.
2. 연구 after_step의 추가 scaler.get_scale()이 step당 GPU→CPU 동기화를 하나 추가했다.
3. 운영 extract_model은 metadata 없는 else에서 embedder_model/speakers_id를 초기화하지 않아 마지막 export가 실패했다. runner가 export까지 요구하므로 completed=false로 남았다. 학습3epoch가 실패했다는 뜻은 아니다.

수정은 연구 주입만 사용한다. train.model_info_path를 원본 metadata의 읽기 경로로 지정하고, exporter의 누락 metadata branch만 프로세스 내부 함수 사본에서 채운다. scale_after는 다음 step의 기존 scale_before로 재구성한다. 새 결과는 ../profile_B_faithful/에 저장하며 이 raw trace·기록은 변경하지 않는다.

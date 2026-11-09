# spawn의 실제 globals를 수정하기 전 진단

연구 hook이 sys.modules의 train 모듈 attribute를 변경했으나 multiprocessing.spawn/runpy의 run 함수는 별도의 원래 globals dictionary를 유지했다. 따라서 첫metadata복구 시도는 실제 run에 적용되지 않았고, 로그는109speakers/contentvec였다. initial_state 수집 wrapper도호출되지않았다. 이 결과 역시목표B의유효프로파일로쓰지않는다.

수정: train.run.__globals__의model_info_path/extract_model/train_and_evaluate에동일한연구주입을적용한다. 실제첫epoch직전에speaker/embedder를assert하고초기G/D SHA를기록한다. 세번째수집결과는../profile_B_target/이며, 기존결과는덮어쓰지않는다.

import participants_functions_t123
#participants_functions.main()

#participant vr files needs to be added to data/vr
participant_sessions=[
        ('EK_207',1, 2), ('EK_207',2, 1), 
        ('EK_240',1, 1), ('EK_240',2, 2),
        ('EK_259', 1,1), ('EK_259',2,2),
        #('EK_260',1,2),
        #('EK_303',1, 2),
        ('EK_352',1,2), ('EK_352', 2,1),
        ('EK_382',1,2), ('EK_382',2,1),
        ('EK_401',1,1), ('EK_401',2,2),
        #('EK_418',1,1), ('EK_418',2,2), #ek 418 don't have timestamps in the 2nd session
        #('EK_424',1,2),
        ('EK_478',1,2), ('EK_478',2,1),
        ('EK_490',1,1), ('EK_490',2,2),
        ('EK_497',1,1), ('EK_497',2,2),
        ('EK_578',1,1), ('EK_578',2,2),
        #('EK_645',1,2), ('EK_645',2,1), #noisy ecg
        #('EK_648', 1,1),
        ('EK_663',1,2), ('EK_663',2,1),
        #('EK_711',1,1),
        #('EK_715',1,1),
        ('EK_716',1,2), ('EK_716',2,1),
        ('EK_753',1,2), ('EK_753',2,1),
        #('EK_795',1,1),
        ('EK_845',1,1), ('EK_845',2,2),
        ('EK_851',1,1), ('EK_851',2,2),
        ('EK_855',1,2), ('EK_855',2,1),
        ('EK_879',1,1), ('EK_879',2,2),
        ('EK_882',1,2), ('EK_882',2,1),
        #('EK_899',1,1),
        ('EK_945',1,2), ('EK_945',2,1),
        #('EK_951',1,1),
        #('EK_993',1,2), ('EK_993',2,1) #noisy ecg
    ]


# participant_sessions=[
#         ('EK_207',1, 1), ('EK_207',2), ('EK_240',1), ('EK_240',2), ('EK_260',1), ('EK_352',1), ('EK_352', 2),
#         ('EK_382',1), ('EK_382',2),  ('EK_401',1), ('EK_401',2), ('EK_478',1), ('EK_478',2), ('EK_490',1), ('EK_490',2),
#         ('EK_497',1), ('EK_497',2), ('EK_578',1), ('EK_578',2), ('EK_648', 1), ('EK_711',1), ('EK_845',1), ('EK_845',2),
#         ('EK_851',1), ('EK_851',2), ('EK_855',1), ('EK_882',1), ('EK_882',2), ('EK_945',1), ('EK_951',2), ('EK_993',1)
#     ]

# participant_sessions = [
#         (845, 1, 1),
#         (478, 2, 1),
#         (578, 2, 1),
#         (352, 1, 1),
#         (401, 1, 1),
#         (2, 1, 1),
#         (382, 2, 1),
#         (490, 1, 1),
#         (6463, 1, 1),
#         (240, 1, 1),
#         (497,1, 1),
#         (851, 1, 1)
#     ]


segment_length = 30
step_size = 30
filename_prefix = '28052025_segment_data_t123'

participants_functions_t123.main(segment_length=segment_length,
                             step_size=step_size,
                             participant_sessions=participant_sessions,
                               filename_prefix=filename_prefix)
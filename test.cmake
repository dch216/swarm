# Use ${SAMPLES} if you need to reference any sample input or configuration file

ENABLE_TESTING()

ADD_TEST(NAME basic COMMAND swarm integrate --defaults )

ADD_TEST(NAME CPU COMMAND swarm test -c ${SAMPLES}/hermite_cpu.cfg -I ${SAMPLES}/smalltest.in.txt -O ${SAMPLES}/smalltest.out.txt )

MACRO(TEST_INTEGRATOR title cfgname)
	ADD_TEST(NAME Verify_${title}
		COMMAND swarm test -c ${SAMPLES}/${cfgname} -I ${SAMPLES}/test.in.txt -O ${SAMPLES}/test.out.txt -v 100) 


TEST_INTEGRATOR(Hermite_GPU hermite.cfg)
TEST_INTEGRATOR(Hermite_adap_GPU hermite_adap.cfg)
TEST_INTEGRATOR(Runge_Kutta_Fixed_Time_Step rkckf.cfg)
TEST_INTEGRATOR(Runge_Kutta_Adaptive_Time_Step rkcka.cfg)
TEST_INTEGRATOR(Euler euler.cfg)

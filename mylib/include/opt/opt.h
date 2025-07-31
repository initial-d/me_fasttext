#ifndef OPT_H_
#define OPT_H_

#ifdef __cplusplus
extern "C" {
#endif

extern const char* opt_param;

typedef struct option
{
    int			cmd_short;
    const char*	cmd_long;
    int			has_arg;
    const char*	arg_name;
    const char*	desc;
} option_t;

int			opt_show_usage(const option_t* opts, const char* desc_str, const char* prog_name, const char* report_addr);
void		opt_show_error(const option_t* opts, int opt);
int			opt_get_index(int argc, char** argv, const option_t* opts);
const char*	opt_get_value(int argc, char** argv, const option_t* opts, char cmd_short, char* cmd_long);

#ifdef __cplusplus
}
#endif

#endif
